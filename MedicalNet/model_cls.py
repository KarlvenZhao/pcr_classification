import os
import torch
from torch import nn
from models import resnet_cls


def _load_pretrain(path):
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError("Pretrain not found: {}".format(path))
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def generate_model(opt):
    assert opt.model in ['resnet']
    assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

    if opt.model_depth == 10:
        model = resnet_cls.resnet10(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)
    elif opt.model_depth == 18:
        model = resnet_cls.resnet18(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)
    elif opt.model_depth == 34:
        model = resnet_cls.resnet34(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)
    elif opt.model_depth == 50:
        model = resnet_cls.resnet50(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)
    elif opt.model_depth == 101:
        model = resnet_cls.resnet101(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)
    elif opt.model_depth == 152:
        model = resnet_cls.resnet152(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)
    else:
        model = resnet_cls.resnet200(
            sample_input_W=opt.input_W,
            sample_input_H=opt.input_H,
            sample_input_D=opt.input_D,
            shortcut_type=opt.resnet_shortcut,
            no_cuda=opt.no_cuda,
            num_classes=opt.num_classes)

    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    if opt.phase != 'test' and opt.pretrain_path:
        pretrain_state = _load_pretrain(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain_state.items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            if pname.find(opt.new_layer_name) >= 0:
                new_parameters.append(p)

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {
            'base_parameters': base_parameters,
            'new_parameters': new_parameters,
        }
        return model, parameters

    return model, model.parameters()
