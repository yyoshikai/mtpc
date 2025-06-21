for sname in 25061815_main_0; do
    python barlowtwins.py --sname $sname --epoch 30
done

<<hist

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