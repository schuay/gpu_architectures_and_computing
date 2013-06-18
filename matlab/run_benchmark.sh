#!/bin/bash

# matlab binary
MATLAB_BIN="matlab"

BENCH_BIN="build/src/stl_bench"

function usage {
    echo $(basename $0)": run the benchmark test with breach and the gpu implementation"
    echo ""
    echo " usage: "
    echo "   -a           run all defined test cases"
    echo "   -t testcase  run only the given testcase (can be given multiple times)"
    echo "   -m dir       directory containing the matlab tests"
    echo "                can also be defined via MATLABTESTS_PATH env. variable"
    echo "   -b dir       path to Breach directory"
    echo "                can also be defined via BREACH_PATH env. variable"
    echo "   -g file      stl_bench executable (default: $BENCH_BIN)"
    echo "                can also be defined via BENCH_BIN environment variable"
    echo "   -c           compare both result files"
    echo "   -n           do not write files, instead use the existing ones"
    echo "   -r           remove files after test case execution"
    echo "   -i num       number of iterations per test case"
    echo "                Breach is significant slower when executing it the first "
    echo "                time. so increase this number to run the test case more times"
    echo "   -o           output report CSV file to matlab/benchmarks/ directory"
    echo "                filename schema: bm_result_<hostname>_YYYYMMDD-HHMM.csv"
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
DO_REPORT=false
while getopts "hat:m:b:g:cnri:o" opt ; do
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
            BENCH_BIN=$OPTARG
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

        o)
            DO_REPORT=true
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

# check for stl_bench binary
[ -x "$BENCH_BIN" ] || 
    failure "error. could not find stl_bench binary. please specify it via -g switch or with BENCH_BIN environment varaible"
BENCH_BIN=$(readlink -f $BENCH_BIN)

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
    failure "error. could not find matlab script 'benchmark.m'. please give correct path to matlabtest directory with -m switch."

# check for testcase definition file
TESTCASES_FILE="$MATLABTESTS_PATH/benchmarks/testcases"
[ -r "$TESTCASES_FILE" ] || 
    failure "error. could not find testcase definition file. expected it in $TESTCASES_FILE" 


# select testcases
if [ ! -z "$TEST_CASES" ] ; then
    # remove leading ' ', replace ' ' by '\|' and replace '[', ']' to '\[', '\]'
    grep_arg=$(echo $TEST_CASES | sed -e 's/^ //' -e 's/ /\\|/g' -e 's/\]/\\\]/g;s/\[/\\\[/g')
    TEST_CASES=$(grep -v '^#' $TESTCASES_FILE | grep ":\($grep_arg\)\$")

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

# temp file for time parsing
out_file=$(mktemp /tmp/$(basename $0).XXXXXXX) || failure "could not create tmp file"

if $DO_REPORT ; then
    REPORT_FILE="$MATLABTESTS_PATH/benchmarks/bm_result_$(hostname)_$(date +'%Y%m%d_%H%M').csv"
    echo "TESTCASE;BREACH;STL_BENCH" > $REPORT_FILE
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
        matlab_cmd="$matlab_cmd r = benchmark('$tcname', '$operator');"
        iter=$(($iter - 1))
    done
    if $DO_NOT_WRITE && [ $s_count -ge 1 -a $b_count -ge 1 ] ; then
        matlab_cmd="$matlab_cmd r = benchmark('$tcname', '$operator');"
    else
        matlab_cmd="$matlab_cmd r = benchmark('$tcname', '$operator', '$test_filename', '$test_filename');"
    fi
    matlab_cmd="$matlab_cmd exit(r)"
    
    $MATLAB_BIN -nosplash -nodesktop -nojvm -r "$matlab_cmd" | tee $out_file | tail -n +10
    result=$?

    [ $result -eq 0 ] ||
        failure "error: test case $tc not defined. exiting"

    breach_mean_time=$(awk 'BEGIN{i=0; o=0;} $1 ~ /Breach:/ {i++; o += $6; } END{if (i > 0) print o/i; else print -1; }' $out_file)
    echo "Breach: resulting mean time: $breach_mean_time s"

    stl_bench_arg=""
    if ! $DO_NOT_WRITE ; then
        stl_bench_arg="$stl_bench_arg -o ${test_filename}.gpuac.trace"
    fi
    if $DO_COMPARE ; then
        stl_bench_arg="$stl_bench_arg -c ${test_filename}.breach.trace"
    fi

    $BENCH_BIN $stl_bench_arg $operator ${test_filename}_sig*.trace | tee $out_file
    result=$?

    [ $result -eq 0 ] ||
        failure "error: stl_bench execution was not successful"

    # now do the multiple gpu iterations
    iter=$(($TC_ITERATIONS - 1))
    while [ $iter -gt 0 ] ; do
        iter=$(($iter - 1))
        $BENCH_BIN $operator ${test_filename}_sig*.trace | tee -a $out_file
        [ $? -eq 0 ] || break
    done

    bench_mean_time=$(awk 'BEGIN{i=0; o=0;} $1 ~ /stl_bench:/ {i++; o += $7; } END{if (i > 0) print o/i; else print -1; }' $out_file)
    echo "stl_bench: resulting mean time: $bench_mean_time s"

    if $DO_REMOVE ; then
        rm -f ${test_filename}_sig*.trace 
        rm -f ${test_filename}.breach.trace 
        rm -f ${test_filename}.gpuac.trace
    fi

    if $DO_REPORT ; then
        echo "$tcname;$breach_mean_time;$bench_mean_time" >> $REPORT_FILE
    fi

    echo 

done

rm -f $out_file

if $DO_REPORT ; then
    echo 
    echo "written results to report file: $REPORT_FILE"
    echo
fi

# change back to old directory
if $DO_POPD ; then
    popd >/dev/null
fi
