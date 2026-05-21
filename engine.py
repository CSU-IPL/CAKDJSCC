from random import choice

from IFKD.utils.IBLoss import IBLoss, CIBLoss
from IFKD.utils.util import *
from IFKD.utils.distortion import *
from IFKD.utils.lr_sched import adjust_learning_rate
from IFKD.utils.CKA import centered_kernel_alignment


def train_one_epoch(args, model_a, model_s, model_t, projector_list, train_loader, optimizer, epoch, logger):
    model_a.encoder.train()
    model_a.decoder.eval()
    model_s.train()
    model_t.eval()
    projector_list.train()

    elapsed, losses, loss_Gs, loss_CIBs, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(8)]
    metrics = [elapsed, losses, loss_Gs, loss_CIBs, psnrs, msssims, cbrs, snrs]
    # CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(args.device)

    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    chan_param = choice(multiple_snr)

    for batch_idx, input in enumerate(train_loader):
        cur_lr = adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)
        start_time = time.time()
        args.global_step += 1
        input = input.to(args.device)
        if args.given_snr:
            recon_image, CBR, SNR, mse, loss_G = model_s(input, chan_param)
            r_t = model_t(input, chan_param)
        else:
            recon_image, CBR, SNR, mse, loss_G = model_s(input)
            r_t = model_t(input)

        fs_list = model_s.encoder.get_feature('features')
        fs_list = [fs_list[j].detach() for j in range(len(fs_list))]
        ft_list = model_t.encoder.get_feature('features')
        ft_list = [ft_list[j].detach() for j in range(len(ft_list))]
        f_s = model_s.encoder.get_feature('enc_output')
        f_s_ = projector_list['projector'](f_s)
        is_conv_features = (f_s_.dim() == 4)
        if is_conv_features:
            B, C, H, W = f_s_.shape
            reshape = lambda v: v.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            B, L, C = f_s_.shape
            reshape = lambda v: v

        if args.given_snr:
            r_a = model_a(input, fs_list, ft_list, chan_param)
        else:
            r_a = model_a(input, fs_list, ft_list)

        f_at = model_a.encoder.get_feature('enc_output').detach()

        loss_CIB = CIBLoss(temp=args.temp, alpha=args.alpha, beta=args.beta, device=args.device)(
            S=F.normalize(reshape(f_s_).mean(dim=1)),
            T=F.normalize(reshape(f_at).mean(dim=1)),
            S_=F.normalize(f_s.flatten(start_dim=1, end_dim=-1)),
            X=F.normalize(input.flatten(start_dim=1, end_dim=-1)),
            Y=F.normalize(reshape(projector_list['projector_x'](r_a[0])).mean(dim=1)),
            g_nce=projector_list['g_nce'],
            g_bo=projector_list['g_bo'],
        )

        dist = torch.nn.MSELoss(reduction='none')
        loss_kd = torch.mean(dist(F.normalize(f_s_, p=2, dim=1) / 0.02, F.normalize(f_at, p=2, dim=1) / 0.02))

        loss = loss_G + args.v * r_a[-1] + args.gamma * (loss_CIB + loss_kd)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.)
        optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        loss_Gs.update(loss_G.item())
        loss_CIBs.update(loss_CIB.item())
        cbrs.update(CBR)
        snrs.update(SNR)
        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())
            # msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            # msssims.update(msssim)
        else:
            psnrs.update(100)
            # msssims.update(100)

        if (args.global_step % args.print_step) == 0:
            process = (args.global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{args.global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.avg:.3f}',
                f'LossG {loss_Gs.avg:.3f}',
                f'LossCIB {loss_CIBs.avg:.3f}',
                f'PSNR {psnrs.avg:.3f}',
                f'lr {cur_lr:.3e}'
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()
    for i in metrics:
        i.clear()


# optimizer.param_groups[0]["lr"]

def test(args, model, test_loader, logger):
    model.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    # results_msssim = np.zeros(len(multiple_snr))
    # CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(args.device)
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            for batch_idx, input in enumerate(test_loader):
                start_time = time.time()
                input = input.to(args.device)
                recon_image, CBR, SNR, mse, loss_G = model(input, SNR)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    # msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    # msssims.update(msssim)
                else:
                    psnrs.update(100)
                    # msssims.update(100)

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})'
                ]))
                logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        # results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")
