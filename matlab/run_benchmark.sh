#!/bin/bash

# matlab binary
MATLAB_BIN="matlab"

GPUAC_BIN="build/src/gpuac"

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
    echo "   -g file      gpuac executable (default: $GPUAC_BIN)"
    echo "                can alos be defined via GPUAC_BIN environmen variable"
    echo "   -c           compare both result files"
    echo "   -n           do not write files, instead use the existing ones"
    echo "   -r           remove files after test case execution"
    echo "   -h           show help (this page)"
    echo ""
}



DO_ALL=false
TEST_CASES=""
DO_COMPARE=false
DO_NOT_WRITE=false
DO_REMOVE=false
while getopts "hat:m:b:g:cnr" opt ; do
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

        g)
            GPUAC_BIN=$OPTARG
            ;;

        c)
            DO_COMPARE=true
            ;;

        n)
            DO_NOT_WRITE=true
            ;;

        r)
            DO_REMOVE=true
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

# check for gpuac binary
if [ ! -x "$GPUAC_BIN" ] ; then
    echo "error. could not find gpuac binary." >&2
    echo "please specify it via -g switch or with GPUAC_BIN environment varaible" >&2
    exit 1;
fi
GPUAC_BIN=$(readlink -f $GPUAC_BIN)

# change to matlab testing path
if [ ! -z "$MATLABTESTS_PATH" ] ; then
    pushd $MATLABTESTS_PATH >/dev/null
    DO_POPD=true
else
    MATLABTESTS_PATH=.
    DO_POPD=false
fi
MATLABTESTS_PATH=$(readlink -f .)

# check for matlab tests path
if [ ! -r "benchmarks/benchmark.m" ] ; then
    echo "error. could not find matlab script 'benchmark.m'. " >&2
    echo "       please give correct path to matlabtest direcory with -m switch." >&2
    if $DO_POPD ; then
        popd >/dev/null
    fi
    exit 1
fi

# check for testcase definition file
TESTCASES_FILE="$MATLABTESTS_PATH/benchmarks/testcases"
if [ ! -r "$TESTCASES_FILE" ] ; then
    echo "error. could not find testcase definition file." >&2
    echo "expacted it in $TESTCASES_FILE" >&2
    if $DO_POPD ; then
        popd > /dev/null
    fi
    exit 1
fi


# select testcases
if [ ! -z "$TEST_CASES" ] ; then
    # remove leading ' ' and replace ' ' by '\|'
    grep_arg=$(echo $TEST_CASES | sed 's/^ //' | sed 's/ /\\|/g')
    TEST_CASES=$(grep ":\($grep_arg\)\$" $TESTCASES_FILE)

elif $DO_ALL ; then
    TEST_CASES=$(cat $TESTCASES_FILE)

else
    echo "no test cases defined or found"
fi

#echo $TEST_CASES
#TEST_CASES=""

TRACES_PATH="$MATLABTESTS_PATH/benchmarks/traces"

# test cases loop
for tc in $TEST_CASES ; do
    operator=${tc%%:*}
    tcname=${tc#*:}

    echo "Testcase $tcname:"

    test_filename="$TRACES_PATH/$tcname"

    matlab_cmd="loadenv;"
    if $DO_NOT_WRITE ; then
        matlab_cmd="$matlab_cmd r = benchmark('$tcname');"
    else
        matlab_cmd="$matlab_cmd r = benchmark('$tcname', '$test_filename', '$test_filename');"
    fi
    matlab_cmd="$matlab_cmd exit(r)"
    
    echo -n "  "
    $MATLAB_BIN -nosplash -nodesktop -nojvm -r "$matlab_cmd" | tail -n +12
    result=$?

    if [ $result -ne 0 ] ; then
        echo "error: test case $tc not defined. exiting" >&2
        break
    fi

    gpuac_arg=""
    if ! $DO_NOT_WRITE ; then
        gpuac_arg="$gpuac_arg -o ${test_filename}.gpuac.trace"
    fi
    if $DO_COMPARE ; then
        gpuac_arg="$gpuac_arg -c ${test_filename}.breach.trace"
    fi

    echo -n "  "
    $GPUAC_BIN $gpuac_arg $operator ${test_filename}_sig*.trace
    result=$?

    if [ $result -ne 0 ] ; then
        echo "error: gpuac execution was not successfull" >&2
        break
    fi

    if $DO_REMOVE ; then
        rm -f ${test_filename}_sig*.trace 
        rm -f ${test_name}.breach.trace 
        rm -f ${test_name}.gpuac.trace
    fi

    echo 

done



# change back to old directory
if $DO_POPD ; then
    popd >/dev/null
fi
