
model_paths=(
    "your model path here"
)




data_dir="./samples"
output_dir="./results"

# Step 1: Trajectory-Level Attribution
for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path")

    echo "=============================================="
    echo "Running model: $model_name"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="


    python component_attri.py \
        --model_id "$model_path" \
        --data_dir $data_dir \
        --output_dir $output_dir

    echo "Finished model: $model_name"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
done



# Step 2: Sentence-Level Attribution & Step 3: Generate Visualization HTML
for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path")

    for f in "$data_dir"/*; do
        [ -f "$f" ] || continue           
        case_file_name="$(basename "$f")"           
        case_tag="${case_file_name%.*}" 

        attr_file="results/${case_tag}_${model_name}_attr_trajectory.json"
        traj_file="${data_dir}/${case_file_name}"
        sentence_attr_output_file="results/${case_tag}_${model_name}_attr_sentence.json"

        echo "=============================================="
        echo "Running model: $model_name"
        echo "Running traj_file: $traj_file"
        echo "Running attr_file: $attr_file"
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "----------------------------------------------"

        python sentence_attri.py \
            --model_id "$model_path" \
            --attr_file "$attr_file" \
            --traj_file "$traj_file" \
            --output_file "$sentence_attr_output_file" \
            --top_k 3


        output_file="results/${case_tag}_${model_name}_all_attr_heatmap.html"

        python case_plot_html.py \
            --traj_attr_file $attr_file \
            --original_traj_file $traj_file \
            --sent_attr_file $sentence_attr_output_file \
            --output_file $output_file
        

        echo "----------------------------------------------"
        echo "Finished model: $model_name"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================================="
    done
done

