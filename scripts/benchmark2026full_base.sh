for ((i=1; i<400; i+=5)); do
	JOB_ID=""
	while [[ -z "$JOB_ID" ]]; do
		JOB_ID=$(qsub -v INICIO=$i,FIM=$((i+4)) benchmark2026_base)
		sleep 0.3
	done
	echo $JOB_ID
done

