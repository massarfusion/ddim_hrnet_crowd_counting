import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    if isinstance(x0, list):
        x0, y = x0[0], x0[-1]
        stackUp = True
    else:
        x0 = x0
        stackUp = False
    
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  # beta(↑) -> alpha(↓) -> cumprod_alpha(↓)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()  # xt = sqrt(cumprod_alpha_t) * x0 + sqrt(1 - cumprod_alpha_t) * EPSILON
    
    if stackUp:
        x = torch.cat([x, y], dim=1)
        
    
    output = model(x, t.float())  # The "UNet" takes only x and t, return a 1-channel map as prediction
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
