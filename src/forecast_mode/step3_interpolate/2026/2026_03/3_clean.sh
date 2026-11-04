year=2026
month=03
output_dir=/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/interpoalted_xlsx_files/$year

for day in {01..31};do
	cd ${year}_${month}${day}/
        mv *xlsx $output_dir/
        rm *err
	rm *out
	rm -r __pycache__
	cd ../

done
