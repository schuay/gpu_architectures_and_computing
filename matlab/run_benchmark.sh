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
    echo "   -i num       number of iteration per test case"
    echo "                Breach is significant slower when executing it the first "
    echo "                time. so increas this number to run the test case more times"
    echo "   -h           show help (this page)"
    echo ""
}

function failure() {
    echo $1 >&2
    if $DO_POPD ; then
        popd >/dev/null
    fi
    exit 1
}


DO_POPD=false   # needed for failure if we changed already in maltab dir
DO_ALL=false
TEST_CASES=""
DO_COMPARE=false
DO_NOT_WRITE=false
DO_REMOVE=false
TC_ITERATIONS=1
while getopts "hat:m:b:g:cnri:" opt ; do
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

        i)  
            TC_ITERATIONS=$OPTARG
            if ! [ $TC_ITERATIONS -ge 1 ] 2>/dev/null ; then
                echo "argument of -i '$OPTARG' not a number or less than one" >&2
                usage
                exit 1
            fi
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
[ ! -z "$BREACH_PATH" -a -r "$BREACH_PATH/InitBreach.m" ] ||
    failure "error. could not find Breach. please specify path to Breach with -b switch or by BREACH_PATH env. var."


# export this environment var (is read by matlab script loadenv.m)
export BREACH_PATH=$(readlink -f $BREACH_PATH)

# check for gpuac binary
[ -x "$GPUAC_BIN" ] || 
    failure "error. could not find gpuac binary. please specify it via -g switch or with GPUAC_BIN environment varaible"
GPUAC_BIN=$(readlink -f $GPUAC_BIN)

# change to matlab testing path
if [ ! -z "$MATLABTESTS_PATH" ] ; then
    [ -d "$MATLABTESTS_PATH" ] || 
        failure "path '$MATLABTESTS_PATH' does not exists or is no directory"
    pushd $MATLABTESTS_PATH >/dev/null
    DO_POPD=true
else
    MATLABTESTS_PATH=.
fi
MATLABTESTS_PATH=$(readlink -f .)

# check for matlab tests path
[ -r "benchmarks/benchmark.m" ] || 
    failure "error. could not find matlab script 'benchmark.m'. please give correct path to matlabtest direcory with -m switch."

# check for testcase definition file
TESTCASES_FILE="$MATLABTESTS_PATH/benchmarks/testcases"
[ -r "$TESTCASES_FILE" ] || 
    failure "error. could not find testcase definition file. expacted it in $TESTCASES_FILE" 


# select testcases
if [ ! -z "$TEST_CASES" ] ; then
    # remove leading ' ' and replace ' ' by '\|'
    grep_arg=$(echo $TEST_CASES | sed 's/^ //' | sed 's/ /\\|/g')
    TEST_CASES=$(grep -v '^#' | grep ":\($grep_arg\)\$" $TESTCASES_FILE)

elif $DO_ALL ; then
    TEST_CASES=$(grep -v '^#' $TESTCASES_FILE)

else
    echo "no test cases defined or found"
fi

#echo $TEST_CASES
#TEST_CASES=""

TRACES_PATH="$MATLABTESTS_PATH/benchmarks/traces"
if [ ! -d "$TRACES_PATH" ] ; then
    mkdir -p $TRACES_PATH || failure "error creating directory for trace files"
fi

# test cases loop
for tc in $TEST_CASES ; do
    operator=${tc%%:*}
    tcname=${tc#*:}

    echo "Testcase $tcname:"

    test_filename="$TRACES_PATH/$tcname"

    s_count=$(ls -1 ${test_filename}_sig*.trace 2>/dev/null | wc -l)
    b_count=$(ls -1 ${test_filename}.breach.trace 2>/dev/null | wc -l)

    matlab_cmd="loadenv;"
    iter=$(($TC_ITERATIONS - 1))
    while [ $iter -gt 0 ] ; do
        matlab_cmd="$matlab_cmd r = benchmark('$tcname');"
        iter=$(($iter - 1))
    done
    if $DO_NOT_WRITE && [ $s_count -ge 1 -a $b_count -ge 1 ] ; then
        matlab_cmd="$matlab_cmd r = benchmark('$tcname');"
    else
        matlab_cmd="$matlab_cmd r = benchmark('$tcname', '$test_filename', '$test_filename');"
    fi
    matlab_cmd="$matlab_cmd exit(r)"
    
    $MATLAB_BIN -nosplash -nodesktop -nojvm -r "$matlab_cmd" | tail -n +10
    result=$?

    [ $result -eq 0 ] ||
        failure "error: test case $tc not defined. exiting"

    gpuac_arg=""
    if ! $DO_NOT_WRITE ; then
        gpuac_arg="$gpuac_arg -o ${test_filename}.gpuac.trace"
    fi
    if $DO_COMPARE ; then
        gpuac_arg="$gpuac_arg -c ${test_filename}.breach.trace"
    fi

    $GPUAC_BIN $gpuac_arg $operator ${test_filename}_sig*.trace
    result=$?

    [ $result -eq 0 ] ||
        failure "error: gpuac execution was not successfull"

    # now do the multiple gpu iterations
    iter=$(($TC_ITERATIONS - 1))
    while [ $iter -gt 0 ] ; do
        iter=$(($iter - 1))
        $GPUAC_BIN $operator ${test_filename}_sig*.trace
        [ $? -eq 0 ] || break
    done

    if $DO_REMOVE ; then
        rm -f ${test_filename}_sig*.trace 
        rm -f ${test_filename}.breach.trace 
        rm -f ${test_filename}.gpuac.trace
    fi

    echo 

done



# change back to old directory
if $DO_POPD ; then
    popd >/dev/null
fi
