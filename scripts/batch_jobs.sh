


for i in {0..2}
do

    #echo "Processing chunk $i"
    # Check if the directory exists
    if [ -d "data/singapore/filtered_text/chunk_$i" ]; then
        echo "Chunk $i already exists"
    else
        # If the directory does not exist, submit the job
        echo "Processing chunk $i"
        sbatch scripts/run_job.sh $i
    fi

done
