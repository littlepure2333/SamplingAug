dir_data = "/data/shizun/"

import datetime
today = datetime.datetime.now().strftime('%Y%m%d')

def set_template(args):
    if args.template.find('EDSR_baseline') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.reset = True
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.save = "{}_{}_x{}_lr{}_ps{}".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('EDSR_test') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.reset = True
        args.data_test = 'Set5+Set14+B100+Urban100'
        args.test_only = 'True'
        args.pre_train = "xxx.pt" # replace with the desired model path
        args.save = "{}_{}_x{}_lr{}_ps{}_test".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('EDSR_sa') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "1"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_p192.pt" #
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)

    if args.template.find('EDSR_nms') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "2"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_nms_p192.pt"
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}_nms".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)

    if args.template.find('EDSR_darts') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "3"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_darts_p192.pt"
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}_nms".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)

    if args.template.find('RCAN_baseline') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.reset = True
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.save = "{}_{}_x{}_lr{}_ps{}".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('RCAN_test') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.reset = True
        args.data_test = 'Set5+Set14+B100+Urban100'
        args.test_only = 'True'
        args.pre_train = "xxx.pt" # replace with the desired model path
        args.save = "{}_{}_x{}_lr{}_ps{}_test".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('RCAN_sa') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_p192.pt" #
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)

    if args.template.find('ESPCN_baseline') >= 0:
        args.model = 'ESPCN'
        args.lr = 1e-3
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.reset = True
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.save = "{}_{}_x{}_lr{}_ps{}".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('ESPCN_test') >= 0:
        args.model = 'ESPCN'
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.reset = True
        args.data_test = 'Set5+Set14+B100+Urban100'
        args.test_only = 'True'
        args.pre_train = "xxx.pt" # replace with the desired model path
        args.save = "{}_{}_x{}_lr{}_ps{}_test".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('ESPCN_sa') >= 0:
        args.model = 'ESPCN'
        args.lr = 1e-3
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_p192.pt" #
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)

    if args.template.find('SRCNN_baseline') >= 0:
        args.model = 'SRCNN'
        args.lr = 1e-4
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.reset = True
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.save = "{}_{}_x{}_lr{}_ps{}".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('SRCNN_test') >= 0:
        args.model = 'SRCNN'
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.reset = True
        args.data_test = 'Set5+Set14+B100+Urban100'
        args.test_only = 'True'
        args.pre_train = "xxx.pt" # replace with the desired model path
        args.save = "{}_{}_x{}_lr{}_ps{}_test".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('SRCNN_sa') >= 0:
        args.model = 'SRCNN'
        args.lr = 1e-4
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_p192.pt" #
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)

    if args.template.find('RDN_baseline') >= 0:
        args.model = 'RDN'
        args.G0 = 64
        args.RDNconfig = 'B'
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 8
        args.print_every = 10
        args.reset = True
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.save = "{}_{}_x{}_lr{}_ps{}".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('RDN_test') >= 0:
        args.model = 'SRCNN'
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.reset = True
        args.data_test = 'Set5+Set14+B100+Urban100'
        args.test_only = 'True'
        args.pre_train = "xxx.pt" # replace with the desired model path
        args.save = "{}_{}_x{}_lr{}_ps{}_test".format(today, args.model, args.scale, args.lr, args.patch_size)

    if args.template.find('RDN_sa') >= 0:
        args.model = 'RDN'
        args.G0 = 64
        args.RDNconfig = 'B'
        args.lr = 1e-5
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 8
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PORTION'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_ii_list_p192.pt" #
        args.save = "{}_{}_x{}_lr{}_ps{}_p{}".format(today, args.model, args.scale, args.lr, args.patch_size, args.data_partion)
