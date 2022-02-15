#!/bin/bash

# Building packages and uploading them to a Gemfury repository

# PUBLISH_URL=$GEMFURY_PUSH_URL
echo "Start"
PUBLISH_URL="https://upload.pypi.org/legacy/"

set -e

DIRS="$@"
BASE_DIR=$(pwd)
SETUP="setup.py"

warn() {
    echo "$@" 1>&2
}

die() {
    warn "$@"
    exit 1
}

build() {
    DIR="${1/%\//}"
    echo "Checking directory $DIR"
    cd "$BASE_DIR/$DIR"
    [ ! -e $SETUP ] && warn "No $SETUP file, skipping" && return
    PACKAGE_NAME=$(python $SETUP --fullname)
    echo "Package $PACKAGE_NAME"
    python -m build || die "Building package $PACKAGE_NAME failed"
	twine upload --repository-url "$PUBLISH_URL" dist/* || die "Uploading package $PACKAGE_NAME failed on file dist/$X"
    # for X in $(ls dist)
    # do
        # # curl -F package=@"dist/$X" "$GEMFURY_URL" || die "Uploading package $PACKAGE_NAME failed on file dist/$X"
		
		# curl -F package=@"dist/$X" "$PUBLISH_URL" || die "Uploading package $PACKAGE_NAME failed on file dist/$X"
    # done
}

if [ -n "$DIRS" ]; then
    for dir in $DIRS; do
        build $dir
    done
else
    ls -d */ | while read dir; do
        build $dir
    done
fi
