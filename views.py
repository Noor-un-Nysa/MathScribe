from django.shortcuts import render
import base64
import cv2
import numpy as np
from .Calculate import evaluate
from .Predict import predictExpression


def draw_view(request):
    return render(request, 'calculator/draw.html')

def predict_view(request):
    if request.method == 'POST':
        draw = request.POST.get('url', '')
        if not draw:
            return render(request, 'calculator/results.html', 
                         {'error': 'No drawing data received'})
        
        try:
            # Process image
            draw = draw[21:]  # Remove base64 prefix
            draw_decoded = base64.b64decode(draw)
            image = np.frombuffer(draw_decoded, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            
            # Get predictions
            predictions = predictExpression(255 - image)
            
            # Evaluation
            if predictions:
                result, error = evaluate(predictions)
            else:
                result, error = None, "No symbols detected"
                
            return render(request, 'calculator/results.html', {
                'prediction': predictions,
                'result': result,
                'error': error
            })
            
        except Exception as e:
            return render(request, 'calculator/results.html', {
                'error': f'Processing error: {str(e)}'
            })
    
    return render(request, 'calculator/draw.html')