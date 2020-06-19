output_file=keywords_for_new_job_titles

echo "Getting 100 related words for job titles"
./query_nn_to_csv german_tabbed_vectors.tsv 100 < new_jobs.txt > datasets/${output_file}.csv

echo "Cleaning first model loading lines from output file"
sed -i '' '1,36d' datasets/${output_file}.csv

echo "Replacing extra comma"
sed -i '' 's/, \"/\"/' datasets/${output_file}.csv

echo "Done. The output file can be found in Starspace/datasets/${output_file}.csv"