#!/bin/bash

# matlab binary
MATLAB_BIN="matlab"



function usage {
    echo $(basename $0)": run the benchmark test with breach and the gpu implemtation"
    echo ""
    echo " usage: "
    echo "   -a           run all defined test cases"
    echo "   -t testcase  run only the given testcase (can be given multiple times)"
    echo "   -m dir       direcotry containing the matlab tests"
    echo "                can also be defined via MATLABTESTS_PATH env. variable"
    echo "   -b dir       path to Breach directory"
    echo "                can also be defined via BREACH_PATH env. variable"
    echo "   -h           show help (this page)"
    echo ""
}



DO_ALL=false
TEST_CASES=""
while getopts "hat:m:b:" opt ; do
    case $opt in 
        a)  
            DO_ALL=true
            ;;

        t)  
            TEST_CASES="$TEST_CASES $OPTARG"
            ;;

        m)
            MATLABTESTS_PATH=$OPTARG
            ;;

        b)
            BREACH_PATH=$OPTARG
            ;;

        h)
            usage
            exit 0
            ;;
        
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;

        :)
            echo "Option -$OPTARG requires argument" >&2
            usage
            exit 1
            ;;
    esac
done

# check if argument definition is consistent
if $DO_ALL && [ ! -z "$TEST_CASES" ] ; then
    echo "you can not use -a and -t at the same time." >&2
    usage
    exit 1
fi

# check for breach path
if [ -z "$BREACH_PATH" -o ! -r "$BREACH_PATH/InitBreach.m" ] ; then
    echo "error. could not find Breach." >&2
    echo "please specify path to Breach with -b switch or by BREACH_PATH env. var." >&2
    exit 1
fi
# export this environment var (is read by matlab script loadenv.m)
export BREACH_PATH=$(readlink -f $BREACH_PATH)

# change to matlab testing path
if [ ! -z "$MATLABTESTS_PATH" ] ; then
    pushd $MATLABTESTS_PATH >/dev/null
fi

# check for matlab tests path
if [ ! -r "benchmarks/benchmark.m" ] ; then
    echo "error. could not find matlab script 'benchmark.m'. " >&2
    echo "       please give correct path to matlabtest direcory with -m switch." >&2
    if [ ! -z "$MATLABTESTS_PATH" ] ; then
        popd >/dev/null
    fi
    exit 1
fi

# test cases loop
for tc in $TEST_CASES ; do

    $MATLAB_BIN -nosplash -nodesktop -nojvm -r "loadenv ; r = benchmark('$tc'); exit(r)"
    RESULT=$?

    if [ $RESULT -eq 1 ] ; then
        echo "error: test case $tc not defined. exiting" >&2
        break
    fi

done


# change back to old directory
if [ ! -z "$MATLABTESTS_PATH" ] ; then
    popd >/dev/null
fi
