import sys
from preprocess_iam_words import preprocess_iam_words
from preprocess_iam_lines import preprocess_iam_lines

if len(sys.argv) < 2:
    print("You must have at least one argument for lines or words")
    sys.exit(0)
if sys.argv[1] == "lines":
    preprocess_iam_lines()
elif sys.argv[1] == "words":
    if len(sys.argv) < 3:
        print("For words, you must specify 0 for only lowercase and 1 for all letters")
        sys.exit(0)
    if int(sys.argv[2]) == 0:
        preprocess_iam_words(True)
    elif int(sys.argv[2]) == 1:
        preprocess_iam_words(False)
    else:
        print("For words, second argument must be 0 (for only lowercase) or 1 (for all letters)")
else:
    print("First arguments must be either lines or words")