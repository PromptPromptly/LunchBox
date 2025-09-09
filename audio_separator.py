import argparse

def separate(input_file, output_dir, stems):
    """Separate input_file into specified stems using Spleeter."""
    try:
        from spleeter.separator import Separator
        separator = Separator(f'spleeter:{stems}stems')
        separator.separate_to_file(input_file, output_dir)
    except Exception as exc:
        print(f"Separation failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Separate an audio file into stems.")
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument("-o", "--output", default="separated", help="Directory to save stems")
    parser.add_argument("-s", "--stems", type=int, choices=[2, 4, 5], default=4,
                        help="Number of stems to separate into")
    args = parser.parse_args()
    separate(args.input, args.output, args.stems)

if __name__ == "__main__":
    main()
