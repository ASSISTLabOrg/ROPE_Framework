while getopts "hvf:" flag; do
 case $flag in
   h) # Handle the -h flag
   # Display script help information
   ;;
   v) # Handle the -v flag
   # Enable verbose mode
   ;;
   f) # Handle the -f flag with an argument
   filename=$OPTARG
   # Input data
   ;;
   \?)
   # Handle invalid options
   ;;
 esac
done

#### submit the job
echo "${filename}" | nc localhost 8888 > output.txt