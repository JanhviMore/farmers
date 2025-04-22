import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class FruitYieldDetector:
    def __init__(self, model_path=None):
        """
        Initialize the Fruit Yield Detector with YOLOv8.
        
        Parameters:
        - model_path: Path to a custom-trained YOLOv8 model.
        """
        # Fruit price database (updated with additional fruits)
        self.fruit_prices = {
            'apple': 0.50,
            'banana': 0.30,
            'orange': 0.40,
            'grape': 0.20,
            'mango': 0.75,
            'pineapple': 1.00,
            'strawberry': 0.10,  # per berry
            'watermelon': 2.00,  # per fruit
            'papaya': 0.80
        }

        try:
            # Load YOLOv8 model
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)  # Load custom-trained model
            else:
                self.model = YOLO('yolov8s.pt')  # Use pretrained YOLOv8 model

            # Customize model settings
            self.model.conf = 0.5  # Confidence threshold (higher for stricter detections)
            self.model.iou = 0.6  # IOU threshold for NMS (lower for fewer overlaps)

        except Exception as e:
            raise RuntimeError(f"Error loading YOLOv8 model: {e}")

    def detect_fruits(self, image_path):
        """
        Detect and classify fruits in the given image.
        
        Parameters:
        - image_path: Path to the input image.

        Returns:
        - Dictionary with detection results.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Resize for consistency
        img = cv2.resize(img, (640, 640))
        
        # Perform inference
        results = self.model(img)

        # Convert results to DataFrame
        df = pd.DataFrame(results[0].boxes.data.cpu().numpy(), 
                          columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
        df['name'] = [results[0].names[int(cls)] for cls in df['class']]

        # Filter detections by confidence score
        df = df[df['confidence'] > 0.5]

        # Count fruits and calculate estimated value
        fruit_counts = df['name'].value_counts()
        total_value = sum(
            self.fruit_prices.get(fruit, 0.50) * count
            for fruit, count in fruit_counts.items()
        )

        # Visualize results
        self._visualize_results(img, df)

        return {
            'detection_results': df,
            'fruit_counts': dict(fruit_counts),
            'total_fruits': len(df),
            'estimated_value': round(total_value, 2),
            'production_cost_estimate': self._calculate_production_cost(fruit_counts)
        }

    def _calculate_production_cost(self, fruit_counts):
        """
        Estimate production costs based on fruit types and quantities.
        """
        production_costs = {
            'apple': 0.20,
            'banana': 0.15,
            'orange': 0.25,
            'grape': 0.10,
            'mango': 0.30,
            'pineapple': 0.50,
            'strawberry': 0.05,
            'watermelon': 1.00,
            'papaya': 0.40
        }
        total_production_cost = sum(
            production_costs.get(fruit, 0.20) * count
            for fruit, count in fruit_counts.items()
        )
        return round(total_production_cost, 2)

    def _visualize_results(self, img, df):
        """
        Visualize detection results on the image.
        """
        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Detected Fruits", img)
        cv2.waitKey(0)  # Wait for key press
        cv2.destroyAllWindows()  # Close the window

def main():
    image_path = "hgic_fruit_tree_apple_1200x625.jpg"  # Update with the path to your image

    # Initialize detector
    detector = FruitYieldDetector()

    # Detect fruits
    results = detector.detect_fruits(image_path)

    # Print results
    print("\n--- Detection Results ---")
    print(f"Total Fruits Detected: {results['total_fruits']}")
    print("Fruit Breakdown:")
    for fruit, count in results['fruit_counts'].items():
        print(f"{fruit.capitalize()}: {count}")
    print(f"\nEstimated Value: ${results['estimated_value']}")
    print(f"Production Cost Estimate: ${results['production_cost_estimate']}")

if __name__ == "__main__":
    main()