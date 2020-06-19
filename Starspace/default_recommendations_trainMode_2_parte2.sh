# MODEL_VERSION=v02

# MODEL_NAME=tab_separated_descriptions_spaced_${MODEL_VERSION}

# TRAIN_FILE_NAME=${MODEL_NAME}.txt
# BASEDOC_PATH=complete_descriptions_spaced_${MODEL_VERSION}.txt
# LABELS_FILE_PATH=labels_for_${MODEL_NAME}.txt

# echo "From data saved in Starspace/datasets"

# echo "Training model"

# cd ./Starspace/

# ./starspace train \
#     -trainFile datasets/"${TRAIN_FILE_NAME}" \
#     -model models/"${MODEL_NAME}" \
#     # -initModel german_tabbed_vectors.tsv \
#     -trainMode 2 \
#     -verbose true \
#     -dim 300 \
#     -thread 40 \
#     -epoch 15 \
#     -trainWord true \
#     -wordWeight 0.7
#     -fileFormat labelDoc
# echo "Making predictions"

# total_placements=$(cat ${BASEDOC_PATH} | wc -l)

# ./query_predict_id_to_file models/"${MODEL_NAME}" $total_placements datasets/"${BASEDOC_PATH}" < Bankkaufmann Krankenschwester Busfahrer Elektrotechniker end > datasets/default_recomendations_${MODEL_VERSION}.txt

# sed -e '1,34d' datasets/default_recomendations_${MODEL_VERSION}.txt > _tmp_file.txt
# sed -e '1d' _tmp_file.txt > datasets/default_recomendations_${MODEL_VERSION}.txt

# ./embed_doc_file models/"${MODEL_NAME}" datasets/"${BASEDOC_PATH}" > models/placements_vectors_trainMode2_${MODEL_VERSION}.tsv

# sed -e '1,8d' models/placements_vectors_trainMode2_${MODEL_VERSION}.tsv > _tmp_file.txt
# sed -e '1d' _tmp_file.txt > models/placements_vectors_trainMode2_${MODEL_VERSION}.tsv

############## To paste in CL ##############


# ./query_predict_id_to_file models/tab_separated_descriptions_spaced_v02_extended 9003 datasets/tab_separated_descriptions_spaced_v02_extended.txt < extended_default_jobs.txt > datasets/default_recommendations_v02_extended.txt

# sed -i '' '1,35d' datasets/default_recommendations_v02_extended.txt

./embed_doc_file models/tab_separated_descriptions_spaced_v02_extended datasets/tab_separated_descriptions_spaced_v02_extended.txt > models/placements_vectors_trainMode2_v02_extended.tsv

sed -i '' '1,3d' models/placements_vectors_trainMode2_v02_extended.tsv