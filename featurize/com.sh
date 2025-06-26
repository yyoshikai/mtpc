
for sname in mtpc_vicregl01 in_sv_mtpc_vicregl01; do
    python VICRegL.py --sname $sname
done
<<hist

# Data Ablations
## VICRegL
for sname in main30_lr01 in_bt_mtpc_vicregl01 in_vicreg_mtpc_vicregl01 in_vicregl_mtpc_vicregl01 tg_bt_mtpc_vicregl01 tg_vicreg_mtpc_vicregl01 tg_vicreg01_mtpc_vicregl01 tg_vicregl01_mtpc_vicregl01; do
    python VICRegL.py --sname $sname
done

## vicreg
for sname in main30 main30_lr01 mtpc_vic in_sv_mtpc_vic in_bt_mtpc_vic in_vic_mtpc_vic in_vicl_mtpc_vic tg_bt_mtpc_vic tg_vic_mtpc_vic tg_vic01_mtpc_vic01 tg_vicl01_mtpc_vic in_vic; do
    python vicreg.py --sname $sname --epoch 30
done

## barlowtwins
for sname in mtpc_bt_0 in_sv_mtpc_bt_0 in_bt_mtpc_bt_0 in_vicreg_mtpc_bt_0 in_vicregl_mtpc_bt_0 tg_bt_mtpc_bt_0 tg_vicreg_mtpc_bt_0 tg_vicreg01_mtpc_bt_0 tg_vicregl01_mtpc_bt_0 in_bt 25061815_main_0; do
    python barlowtwins.py --sname $sname --epoch 30
done


# 250620 batch size
python barlowtwins.py --sname 250620_bsz/64_adam_0 --epoch 30
python barlowtwins.py --sname 250620_bsz/64_adam_128_0 --epoch 30
python barlowtwins.py --sname 250620_bsz/64_lars_0 --epoch 30
python barlowtwins.py --sname 250620_bsz/256_adam_0 --epoch 30
python barlowtwins.py --sname 250620_bsz/256_lars_0 --epoch 30
python barlowtwins.py --sname 250620_bsz/512_adam_0 --epoch 30
python barlowtwins.py --sname 250620_bsz/512_lars_0 --epoch 30


for sname in 25061719_mid13_0 25061719_mid14_0; do
    python barlowtwins.py --sname $sname --weight checkpoint_latest
done
for sname in 25061815_mid15_0 25061814_mid16_0 25061814_mid17_0; do
    python barlowtwins.py --sname $sname --epoch 100
done
hist