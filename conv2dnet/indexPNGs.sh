#!/bin/bash

index_dir() {
	counter=0
	for file in ./rockpaperscissors/${1}/${2}/*.png
	do
		mv "${file}" ./rockpaperscissors/${1}/${2}/${2}_${counter}.png
		(( counter+=1 ))
	done
}

# Args:
# $1 - name,
# $2 - train count,
# $3 - test count,
# $4 - total count
split_data() {
	# move the test files
	mapfile -t files < <(find "./rockpaperscissors/${1}" -type f)

	mkdir -p "./rockpaperscissors/train/${1}"
	for(( i=0; i<${2}; i++ )); do
		mv "${files[$i]}" "./rockpaperscissors/train/${1}/"
	done

	# move the train files
	mkdir -p "./rockpaperscissors/test/${1}"
	for(( i=${2}; i<$((${3} + ${2})); i++ )); do
		mv "${files[$i]}" "./rockpaperscissors/test/${1}/"
	done
	
	# move the validation files
	mkdir -p "./rockpaperscissors/validation/${1}"
	for(( i=$(( ${2} + ${3} )); i<${4}; i++ )); do
		mv "${files[$i]}" "./rockpaperscissors/validation/${1}"
	done
}

scissors_count=$(ls ./rockpaperscissors/scissors -iq | wc -l)
rock_count=$(ls ./rockpaperscissors/rock -iq | wc -l)
paper_count=$(ls ./rockpaperscissors/paper -iq | wc -l)

scissors_train_cnt=$(( ${scissors_count}*80/100 ))
scissors_validation_cnt=$(( ${scissors_count}*10/100))
scissors_test_cnt=$(( ${scissors_count} - ${scissors_validation_cnt} - ${scissors_train_cnt} ))

rock_train_cnt=$(( ${rock_count}*80/100 ))
rock_validation_cnt=$(( ${rock_count}*10/100))
rock_test_cnt=$(( ${rock_count} - ${rock_validation_cnt} - ${rock_train_cnt} ))

paper_train_cnt=$(( ${paper_count}*80/100 ))
paper_validation_cnt=$(( ${paper_count}*10/100))
paper_test_cnt=$(( ${paper_count} - ${paper_validation_cnt} - ${paper_train_cnt} ))

split_data "scissors" $scissors_train_cnt $scissors_test_cnt $scissors_count
split_data "paper" $paper_train_cnt $paper_test_cnt $paper_count
split_data "rock" $rock_train_cnt $rock_test_cnt $rock_count

index_dir "test" "scissors"
index_dir "train" "scissors"
index_dir "validation" "scissors"

index_dir "test" "rock"
index_dir "train" "rock"
index_dir "validation" "rock"

index_dir "test" "paper"
index_dir "train" "paper"
index_dir "validation" "paper"


