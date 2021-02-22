#!/bin/bash

# This script has prerequisite variables:
# BENCHMARK_SCRIPT which is the python3 benchmark script
# BENCHMARK_DB_CONFIG which contains the configuration for the database
# LIQUIBASE_PATH which points to the liquibase sh path
# LIQUIBASE_CLASSPATH which poins to a classpath, including oracle jdbc (example: LIQUIBASE_CLASSPATH=/home/user/ojdbc8.jar:/home/user/secondlib.jar)


WAREHOUSES=40
SCHEMA_USERS=16
LOAD_USERS=64
RAMPUP_TIME=1
PREMIGRATION_TIME=1
POST_MIGRATION_TIME=5

if [ -z "$BENCHMARK_SCRIPT" ]; then
    echo "BENCHMARK_SCRIPT is not set"
    exit 1
fi

if [ ! -f "$BENCHMARK_SCRIPT" ]; then
	echo "file $BENCHMARK_SCRIPT does not exist"
	exit 1
fi

if [ -z "$BENCHMARK_DB_CONFIG" ]; then
    echo "BENCHMARK_DB_CONFIG is not set"
    exit 1
fi

if [ ! -f "$BENCHMARK_DB_CONFIG" ]; then
	echo "file $BENCHMARK_DB_CONFIG does not exist"
	exit 1

fi

if [ -z "$LIQUIBASE_PATH" ]; then
	echo "LIQUIBASE_PATH is not set"
	exit 1
fi

if [ ! -f "$LIQUIBASE_PATH" ]; then
	echo "file $LIQUIBASE_PATH does not exist"
	exit 1
fi

if [ -z "$LIQUIBASE_CLASSPATH" ]; then
	echo "LIQUIBASE_CLASSPATH is not set"
	exit 1
fi

if ! command -v "python3" &> /dev/null
then
    echo "Python3 could not be found"
    exit
fi

if [ -z "$1" ]; then
	echo "No input directory provided"
	exit 1
fi

INPUT_DIR=$1
if [ ! -d "$INPUT_DIR" ]; then
	echo "directory $INPUT_DIR does not exist"
	exit 1
fi

PREREQUISITES_FILE=`find $INPUT_DIR -name prerequisites.* | head -n 1`
COMMAND_SUFFIX=""
if [ ! -z "$PREREQUISITES_FILE " ]; then
	COMMAND_SUFFIX="-p $PREREQUISITES_FILE"
fi

CHANGE_FILE=`find $INPUT_DIR -name change.* | head -n 1`
if [[ -z "$CHANGE_FILE" || ! -f "$CHANGE_FILE" ]]; then
	echo "No change file (change.xml/yml) found in $INPUT_DIR"
	exit 1
fi

CHANGE_NAME=`basename $INPUT_DIR`
BENCHMARK_COMMAND="-v $BENCHMARK_DB_CONFIG -su $SCHEMA_USERS -ks -lu $LOAD_USERS -w $WAREHOUSES -r $RAMPUP_TIME -b $PREMIGRATION_TIME -a $POST_MIGRATION_TIME -n $CHANGE_NAME -o outputs -m $CHANGE_FILE $COMMAND_SUFFIX"
echo $BENCHMARK_COMMAND
$BENCHMARK_SCRIPT $BENCHMARK_COMMAND