import torch
archs = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", archs)
assert torch.cuda.is_available(), "no CUDA device visible"
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
assert "sm_120" in archs, "this torch build has no Blackwell (sm_120) kernels"
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok", float((x @ x).sum()))
print("L1_OK")
