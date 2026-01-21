"""
Train YOLOv8 on Construction-PPE Dataset
AUTO-DETECT GPU or fallback to CPU
"""

if __name__ == '__main__':
    from ultralytics import YOLO
    import torch

    print("="*70)
    print("TRAINING YOLOV8 ON CONSTRUCTION-PPE DATASET")
    print("="*70)

    # Check GPU
    cuda_available = torch.cuda.is_available()
    print(f"\nGPU Available: {cuda_available}")

    if cuda_available:
        device = 0
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Training on: GPU")
    else:
        device = 'cpu'
        print("GPU not available - training on CPU (will be SLOW!)")
        print("Tip: Install PyTorch with CUDA support for faster training")

    # Initialize model
    print("\n[INFO] Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')

    # Training configuration
    print("\n[INFO] Starting training...")
    print("Configuration:")
    print(f"  - Device: {'GPU (cuda:0)' if cuda_available else 'CPU'}")
    print("  - Model: YOLOv8n")
    print("  - Dataset: Construction-PPE")
    print(f"  - Epochs: {'50' if cuda_available else '20 (reduced for CPU)'}")
    print("  - Image size: 640")
    print(f"  - Batch size: {'16' if cuda_available else '8 (reduced for CPU)'}")

    epochs = 50 if cuda_available else 20
    batch = 16 if cuda_available else 8

    # Train
    try:
        results = model.train(
            data='datasets/construction-ppe/data.yaml',
            epochs=epochs,
            imgsz=640,
            batch=batch,
            name='ppe_detector',
            patience=15,
            device=device,
            workers=0,  # 0 for Windows to avoid multiprocessing issues
            plots=True,
            save=True,
            val=True,
            cache=False,
            project='runs/detect',
            exist_ok=True,
            verbose=True
        )
        
        # Validation
        print("\n[INFO] Running validation...")
        metrics = model.val()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best model saved to: runs/detect/ppe_detector/weights/best.pt")
        print(f"\nMetrics:")
        print(f"  mAP50: {metrics.box.map50:.3f}")
        print(f"  mAP50-95: {metrics.box.map:.3f}")
        print("="*70)
        
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        print("Check dataset path and format")
