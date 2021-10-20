for i in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
python scripts/average_checkpoints.py --inputs \
	../checkpoints/large.seed$i \
    --num-epoch-checkpoints 5 \
	--output ../checkpoints/avg$i.pt
done
