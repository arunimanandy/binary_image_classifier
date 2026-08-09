import argparse, requests, sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--url', default='http://localhost:8000'); p.add_argument('--image', required=True); args=p.parse_args()
    health=requests.get(args.url.rstrip('/') + '/health', timeout=10)
    if health.status_code != 200 or health.json().get('status') != 'ok':
        print('Health check failed', health.text); sys.exit(1)
    with open(args.image, 'rb') as f:
        pred=requests.post(args.url.rstrip() + '/predict', files={'file': f}, timeout=30)
    if pred.status_code != 200 or 'label' not in pred.json():
        print('Prediction failed', pred.text); sys.exit(1)
    print('Smoke test passed:', pred.json())
if __name__ == '__main__': main()
