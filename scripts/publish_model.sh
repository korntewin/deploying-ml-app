
#

RELATIVE_DIRS="$@"
SETUPFILE="setup.py"
CUR_DIRS=$(pwd)

# execute warning to stderror
warn() {
    echo "$@" 1>&2
}

# exit the terminal if error
die() {
    warn "$@"
    exit 1
}

# build script
build() {
    cd "$CUR_DIRS/$RELATIVE_DIRS"
    [ ! -e "$SETUPFILE" ] && warn "There is no $SETUPFILE" && return
    PACKAGE_NAME=$(python3 $SETUPFILE --fullname)
    echo "Package $PACKAGE_NAME"
    python3 $SETUPFILE sdist bdist_wheel || die "Building package $PACKAGE_NAME failed"

    for file in `ls dist`;
        do curl -F package=@dist/${file} ${GEM_PUSH_URL} || die "Fail to upload ${file} to Gemfury";
    done;

}

if [ -n "$RELATIVE_DIRS" ]; then
    build $RELATIVE_DIRS
else
    build .
fi