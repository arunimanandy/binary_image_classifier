from PIL import Image
from src.ml.inference import load_model, predict_image


def test_predict_image_returns_label_and_probabilities():
    model = load_model('models/non_existing_unit_test_model.pt')
    img = Image.new('RGB', (224, 224), (120, 120, 120))
    result = predict_image(model, img)
    assert result['label'] in ['cat', 'dog']
    assert set(result['probabilities'].keys()) == {'cat', 'dog'}
