import os, sys, torch, pathlib
print("python:", sys.executable)
print("torch:", torch.__version__)
print("mps:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
binp = pathlib.Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
print("torch_shm_manager:", binp)
print("exists:", binp.exists())
if binp.exists():
    st = binp.stat()
    print("mode(oct):", oct(st.st_mode))
    print("executable:", os.access(binp, os.X_OK))
