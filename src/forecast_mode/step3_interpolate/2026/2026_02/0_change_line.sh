year=2026
month=02
  
for day in {01..28}; do
        dir="${year}_${month}${day}"
        #sed -i "s/dir_month = '${year}_01'/dir_month = '${year}_${month}'/" "$dir"/step1_model_inputs.py
	sed -i "s/obs_init = '20260101_20260201'/obs_init = '20260201_20260301'/" "$dir"/step2_interpolate.py 
done



