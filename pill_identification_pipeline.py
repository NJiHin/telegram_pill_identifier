"""
Pill Identification Pipeline
Implementation based on: "An Accurate Deep Learning-Based System for Automatic
Pill Identification: Model Development and Validation" (Heo et al., 2023)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from rnn_model import Seq2SeqWithAttention


class PillIdentificationPipeline:
    """Complete pill identification pipeline with YOLO, ResNet, and RNN"""

    def __init__(self, yolo_path, resnet_path, rnn_path, database_path, device=None):
        """
        Initialize the pipeline with all three models

        Args:
            yolo_path: Path to YOLO weights
            resnet_path: Path to ResNet checkpoint
            rnn_path: Path to RNN checkpoint
            database_path: Path to pill database CSV
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device(
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_path)

        # Load ResNet model
        print("Loading ResNet model...")
        resnet_checkpoint = torch.load(resnet_path, weights_only=False)
        self.resnet = MultiTaskResNet(
            resnet_checkpoint['num_shape_classes'],
            resnet_checkpoint['num_color_classes']
        ).to(self.device)
        self.resnet.load_state_dict(resnet_checkpoint['model_state_dict'])
        self.resnet.eval()
        self.shape_encoder = resnet_checkpoint['shape_encoder']
        self.color_encoder = resnet_checkpoint['color_encoder']

        # ResNet transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load RNN model
        print("Loading RNN model...")
        rnn_checkpoint = torch.load(rnn_path, weights_only=False)
        self.rnn = Seq2SeqWithAttention(**rnn_checkpoint['model_config']).to(self.device)
        self.rnn.load_state_dict(rnn_checkpoint['model_state_dict'])
        self.rnn.eval()

        # RNN vocab
        self.char_to_idx = rnn_checkpoint['char_to_idx']
        self.idx_to_char = rnn_checkpoint['idx_to_char']
        self.EOS_IDX = rnn_checkpoint['EOS_IDX']

        # Load RNN dataset info for encoding
        dataset = torch.load('rnn_dataset.pt', weights_only=False)
        self.char_encoder = dataset['char_encoder']
        self.shape_encoder_rnn = dataset['shape_encoder']
        self.color_encoder_rnn = dataset['color_encoder']
        self.INPUT_CHARS = dataset['char_encoder'].categories_[0].tolist()
        self.ALL_SHAPES = dataset['ALL_SHAPES']
        self.ALL_COLORS = dataset['ALL_COLORS']

        # Load database
        print("Loading database...")
        self.database = pd.read_csv(database_path)
        print(f"Database loaded: {len(self.database)} pills")

        print("Pipeline ready!")

    def sort_boxes_left_to_right(self, boxes):
        """Sort bounding boxes left-to-right, top-to-bottom"""
        if len(boxes) == 0:
            return []
        centers = [(box['bbox'][0], box['bbox'][1]) for box in boxes]
        sorted_indices = sorted(
            range(len(centers)),
            key=lambda i: (round(centers[i][1] * 10), centers[i][0])
        )
        return [boxes[i] for i in sorted_indices]

    def encode_features_for_rnn(self, yolo_detections, shape, color):
        """
        Encode YOLO detections + ResNet features for RNN input
        Format: [x, y, char_OHE, shape_OHE, color_OHE] per character
        """
        sorted_boxes = self.sort_boxes_left_to_right(yolo_detections)

        # Encode ResNet features (same for all characters)
        shape_ohe = self.shape_encoder_rnn.transform([[shape]])[0]
        color_ohe = self.color_encoder_rnn.transform([[color]])[0]

        sequences = []
        for det in sorted_boxes:
            x_center, y_center, w, h = det['bbox']
            char = det['class_name'].upper()

            # Character one-hot encoding
            char_ohe = self.char_encoder.transform([[char]])[0]

            # Concatenate all features
            feature_vector = np.concatenate([
                [x_center, y_center],
                char_ohe,
                shape_ohe,
                color_ohe
            ])
            sequences.append(feature_vector)

        return np.array(sequences) if sequences else np.array([])

    def create_mask(self, X):
        """Create mask for valid (non-padded) positions"""
        mask = (X.sum(dim=2) != 0).float()
        return mask

    def decode_sequence(self, indices):
        """Convert token indices to string, stopping at EOS"""
        chars = []
        for idx in indices:
            idx_val = idx.item() if torch.is_tensor(idx) else idx
            if idx_val == self.EOS_IDX:
                break
            if idx_val in self.idx_to_char:
                char = self.idx_to_char[idx_val]
                if char not in ['<SOS>', '<PAD>', '<EOS>']:
                    chars.append(char)
        return ''.join(chars)

    def predict_pill(self, image_path):
        """
        Run complete pipeline: YOLO -> ResNet -> RNN

        Returns:
            dict with shape, color, and corrected imprint
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert('RGB')

        # Step 1: YOLO - Detect imprinted characters
        yolo_results = self.yolo.predict(image_path, conf=0.25, verbose=False)[0]
        detections = []
        for box in yolo_results.boxes:
            detections.append({
                'class_id': int(box.cls),
                'class_name': yolo_results.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xywhn.tolist()[0]  # normalized [x_center, y_center, width, height]
            })

        # Step 2: ResNet - Classify shape and color
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            shape_out, color_out = self.resnet(image_tensor)
            pred_shape_idx = shape_out.argmax(dim=1).item()
            pred_color_idx = color_out.argmax(dim=1).item()

        pred_shape = self.shape_encoder.inverse_transform([pred_shape_idx])[0]
        pred_color = self.color_encoder.inverse_transform([pred_color_idx])[0]

        # Step 3: RNN - Correct imprint characters
        corrected_imprint = ""
        if detections:
            # Encode input for RNN
            features = self.encode_features_for_rnn(detections, pred_shape, pred_color)

            if len(features) > 0:
                # Pad to match RNN expected input size (48 characters max)
                X = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # [1, seq_len, 54]

                # Pad if necessary
                max_len = 48
                if X.shape[1] < max_len:
                    padding = torch.zeros(1, max_len - X.shape[1], X.shape[2]).to(self.device)
                    X = torch.cat([X, padding], dim=1)

                # RNN inference
                src_mask = self.create_mask(X).to(self.device)
                with torch.no_grad():
                    predictions, _, lengths = self.rnn.predict(X, max_len=50, src_mask=src_mask)

                # Decode prediction
                corrected_imprint = self.decode_sequence(predictions[0][:lengths[0]].cpu())

        return {
            'shape': pred_shape,
            'color': pred_color,
            'imprint': corrected_imprint,
            'yolo_detections': detections
        }

    def levenshtein_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def count_overlapping_chars(self, s1, s2):
        """Count overlapping characters between two strings"""
        from collections import Counter
        counter1 = Counter(s1.lower())
        counter2 = Counter(s2.lower())
        overlap = sum((counter1 & counter2).values())
        return overlap

    def calculate_similarity(self, prediction, db_pill):
        """
        Calculate similarity score as described in the paper:
        - Feature scores: 1/3 each for shape/color exact match
        - Imprint edit distance (normalized): 0-1
        - Imprint character overlap (normalized): 0-1
        """
        # Feature scores (1/3 each)
        shape_score = 1/3 if prediction['shape'] == db_pill['shape'] else 0
        color_score = 1/3 if prediction['color'] == db_pill['color'] else 0

        # Text similarity (normalized edit distance)
        pred_text = str(prediction['imprint']).upper()
        db_text = str(db_pill['imprint']).upper()

        edit_dist = self.levenshtein_distance(pred_text, db_text)
        total_len = max(len(pred_text), len(db_text))
        edit_score = (1 - edit_dist / total_len) if total_len > 0 else 0

        # Character overlap
        overlap = self.count_overlapping_chars(pred_text, db_text)
        overlap_score = (overlap * 2) / (len(pred_text) + len(db_text)) if (len(pred_text) + len(db_text)) > 0 else 0

        total = shape_score + color_score + edit_score + overlap_score
        return total

    def retrieve_top_k(self, prediction, k=3):
        """
        Retrieve top-k most similar pills from database

        Args:
            prediction: dict with 'shape', 'color', 'imprint'
            k: number of top results to return

        Returns:
            list of tuples: [(pill_info, score), ...]
        """
        scores = []

        for idx in range(len(self.database)):
            pill = self.database.iloc[idx]

            db_pill = {
                'shape': pill['splshape_text'],
                'color': pill['splcolor_text'],
                'imprint': pill['splimprint_clean']
            }

            score = self.calculate_similarity(prediction, db_pill)
            scores.append((pill, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class MultiTaskResNet(nn.Module):
    """ResNet-18 with multitask heads for shape and color"""

    def __init__(self, num_shapes, num_colors):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.shape_head = nn.Linear(in_features, num_shapes)
        self.color_head = nn.Linear(in_features, num_colors)

    def forward(self, x):
        features = self.backbone(x)
        return self.shape_head(features), self.color_head(features)


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PillIdentificationPipeline(
        yolo_path='runs/detect/pill_imprint_v17/weights/best.pt',
        resnet_path='resnet_model/pill_classifier_full.pth',
        rnn_path='checkpoints/best_model.pt',
        database_path='data/pillbox_cleaned_F.csv'
    )

    # Load test images
    with open('main_pipeline_test_image_names.txt') as f:
        test_images = [line.strip() for line in f if line.strip()]

    image_dir = Path('data/pillbox_production_images_full_202008')

    print(f"\n{'='*80}")
    print(f"RUNNING PIPELINE ON {len(test_images)} TEST IMAGES")
    print(f"{'='*80}\n")

    # Process each image
    for i, img_name in enumerate(test_images[:10], 1):  # First 10 for demo
        img_path = image_dir / img_name

        if not img_path.exists():
            print(f"{i}. {img_name} - NOT FOUND")
            continue

        # Run pipeline
        prediction = pipeline.predict_pill(img_path)

        # Retrieve top 3 matches
        top_3 = pipeline.retrieve_top_k(prediction, k=3)

        print(f"{i}. {img_name}")
        print(f"   Predicted: {prediction['shape']}, {prediction['color']}, '{prediction['imprint']}'")
        print(f"   Top 3 matches:")
        for j, (pill, score) in enumerate(top_3, 1):
            print(f"      {j}. {pill['medicine_name'][:50]:50s} (score: {score:.3f})")
        print()
