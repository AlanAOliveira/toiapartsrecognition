class AIPartIdentifier:
    """Use a trained model to visually identify parts (no text needed)."""

    def __init__(self, model_path: str = None, catalog_path: str = None):
        self.model = None
        self.part_catalog = {}

        if model_path:
            self.load_model(model_path)
        if catalog_path:
            self.load_catalog(catalog_path)

    def load_model(self, model_path: str):
        """Load pre-trained classification model."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)

    def load_catalog(self, catalog_path: str):
        """Load part number catalog (JSON mapping)."""
        import json
        with open(catalog_path, 'r') as f:
            self.part_catalog = json.load(f)

    def predict_part(self, image_path: str) -> dict:
        """Predict part number from image using trained model."""
        import tensorflow as tf

        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        part_number = self.part_catalog.get(
            str(predicted_class), "UNKNOWN"
        )

        return {
            "image_path": image_path,
            "predicted_part_number": part_number,
            "confidence": confidence,
            "class_id": int(predicted_class)
        }