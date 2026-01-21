import torch
print("="*60)
print("CUDA VERIFICATION")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print("\n✅ GPU READY FOR TRAINING!")
else:
    print("\n❌ CUDA not available")
print("="*60)
