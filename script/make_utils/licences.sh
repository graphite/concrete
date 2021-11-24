#!/bin/bash

set -e

BASENAME="licences"
LICENCE_DIRECTORY="licences"
CHECK=0
DIFF_TOOL="diff --ignore-all-space --ignore-tab-expansion --ignore-space-change --ignore-all-space --ignore-blank-lines --strip-trailing-cr"
TMP_VENV_PATH="/tmp/tmp_venv"
DO_USER_LICENCES=1
DO_DEV_LICENCES=1
OUTPUT_DIRECTORY="${LICENCE_DIRECTORY}"

while [ -n "$1" ]
do
   case "$1" in
        "--check" )
            CHECK=1
            OUTPUT_DIRECTORY=$(mktemp -d)
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

UNAME=$(uname)
if [ "$UNAME" == "Darwin" ]
then
    OS=mac
elif [ "$UNAME" == "Linux" ]
then
    OS=linux
else
    echo "Problem with OS"
    exit 255
fi

if [ $DO_USER_LICENCES -eq 1 ]
then
    # Licences for user (install in a temporary venv)
    echo "Doing licences for user"

    FILENAME="${BASENAME}_${OS}_user.txt"
    LICENSES_FILENAME="${LICENCE_DIRECTORY}/${FILENAME}"
    NEW_LICENSES_FILENAME="${OUTPUT_DIRECTORY}/${FILENAME}"

    rm -rf $TMP_VENV_PATH/tmp_venv
    python3 -m venv $TMP_VENV_PATH/tmp_venv

    # SC1090: Can't follow non-constant source. Use a directive to specify location.
    # shellcheck disable=SC1090
    source $TMP_VENV_PATH/tmp_venv/bin/activate

    python -m pip install -U pip wheel
    python -m pip install -U --force-reinstall setuptools
    poetry install --no-dev
    python -m pip install pip-licenses
    pip-licenses | grep -v "pkg\-resources\|concretefhe" | tee "${NEW_LICENSES_FILENAME}"
    deactivate

    if [ $CHECK -eq 1 ]
    then
        echo "$DIFF_TOOL $LICENSES_FILENAME ${NEW_LICENSES_FILENAME}"
        $DIFF_TOOL "$LICENSES_FILENAME" "${NEW_LICENSES_FILENAME}"
        echo "Success: no update in $LICENSES_FILENAME"
    fi
fi

if [ $DO_DEV_LICENCES -eq 1 ]
then
    # Licences for developer (install in a temporary venv)
    echo "Doing licences for developper"

    FILENAME="${BASENAME}_${OS}_dev.txt"
    LICENSES_FILENAME="${LICENCE_DIRECTORY}/${FILENAME}"
    NEW_LICENSES_FILENAME="${OUTPUT_DIRECTORY}/${FILENAME}"

    rm -rf $TMP_VENV_PATH/tmp_venv
    python3 -m venv $TMP_VENV_PATH/tmp_venv

    # SC1090: Can't follow non-constant source. Use a directive to specify location.
    # shellcheck disable=SC1090
    source $TMP_VENV_PATH/tmp_venv/bin/activate

    make setup_env
    pip-licenses | grep -v "pkg\-resources\|concretefhe" | tee "${NEW_LICENSES_FILENAME}"
    deactivate

    if [ $CHECK -eq 1 ]
    then

        echo "$DIFF_TOOL $LICENSES_FILENAME ${NEW_LICENSES_FILENAME}"
        $DIFF_TOOL "$LICENSES_FILENAME" "${NEW_LICENSES_FILENAME}"
        echo "Success: no update in $LICENSES_FILENAME"
    fi
fi

rm -f ${LICENCE_DIRECTORY}/licences_*.txt.tmp
rm -rf $TMP_VENV_PATH/tmp_venv

echo "End of licence script"
