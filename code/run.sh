#!/bin/bash
 
if [[ "$#" -ne 4 && "$#" -ne 5 ]]
then
	echo "Error: Incorrect no. of arguments. Expected 4 or 5 arguments."
elif [ "$#" -eq 5 ]
then
	if [ "$1" -ne 1 ]
	then
		echo "Error: Incorrect no. of arguments. Expected exactly 5 arguments only for Q1."
	else
		if [ "$5" = "a" ]
		then
			python question1a.py "$2" "$3" "$4"
		elif [ "$5" = "b" ]
		then 
			python question1b.py "$2" "$3" "$4"
		elif [ "$5" = "c" ]
		then 
			python question1c.py "$2" "$3" "$4"
		elif [ "$5" = "d" ]
		then 
			python question1d.py "$2" "$3" "$4"
		else
			echo "Error: Incorrect part number for Q1. Part number can be a, b, c, or d."
		fi
	fi
else
	if [ "$1" -ne 2 ]
	then
		echo "Error: Incorrect no. of arguments. Expected exactly 4 arguments only for Q2."
	else
		if [ "$4" = "a" ]
		then
			python question2a.py "$2" "$3"
		elif [ "$4" = "b" ]
		then 
			python question2b.py "$2" "$3"
		elif [ "$4" = "c" ]
		then 
			python question2c.py "$2" "$3"
		elif [ "$4" = "d" ]
		then 
			python question2d.py "$2" "$3"
		elif [ "$4" = "e" ]
		then 
			python question2e.py "$2" "$3"
		elif [ "$4" = "f" ]
		then 
			python question2f.py "$2" "$3"
		else
			echo "Error: Incorrect part number for Q2. Part number can be a, b, c, d, e or f."
		fi
	fi
fi

