def lora_backward_hook(lora_module, grad_in, grad_out):
    print("Module_name: ", lora_module)
    print("grad_in[0]_shape: ",grad_in[0].shape)
    print("grad_in[0]: ", grad_in[0])
    print("grad_out[0]_shape: ", grad_out[0].shape)
    print("grad_out[0]: ", grad_out[0])
    print("-------------------------------")