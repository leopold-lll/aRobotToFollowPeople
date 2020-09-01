# USAGE:
# python main.py --detector ssd --tracker csrt --classifier resNet50 --videoOut videosOut/out --videoIn videosIn/shelfy_dataset/v01-basecase.mp4 --showLog --firstPhase 5000 --useGPU
import argparse
from follower import Follower

def parseArguments() -> None:
	"""Construct the argument parser and parse the arguments."""
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--detector",		type=str,	default="ssd",		help="A detector name, choises: yolo or ssd")
	ap.add_argument("-t", "--tracker",		type=str,	default="csrt",		help="A tracker name, choises: kcf or csrt")
	ap.add_argument("-c", "--classifier",	type=str,	default="resNet50",	help="A classifier name, choises: resNet50, googleNet")

	ap.add_argument("-s", "--sourceFPS",	type=int,	default=15,			help="The FPS rate of the given input video.")
	ap.add_argument("-v", "--videoIn",		type=str,	required=True,		help="The path to the input video.")
	ap.add_argument("-o", "--videoOut",		type=str,	default=None,		help="The path to the output video (no extension).")

	ap.add_argument("-g", "--useGPU", action="store_true", default=False,	help="A flag to use a GPU for the DNNs.")
	ap.add_argument("-l", "--showLog",action="store_true", default=False,	help="A flag to use show the code log during execution. Set to False to execute with high performances.")
	ap.add_argument("-f", "--firstPhase",	type=int,	default=5000,		help="Lenght in millisecond of the first phase (learn the subject).")
	args = vars(ap.parse_args())
	return(args)


def main() -> None:
	args = parseArguments()

	follower = Follower(
		args["detector"], args["tracker"], args["classifier"], 
		args["sourceFPS"], args["videoIn"], args["videoOut"], 
		args["firstPhase"], args["useGPU"], args["showLog"])

	while True:
		end = follower.follow()
		if end is None:
			break
		else:
			print("Subject position:", end)

if __name__=="__main__":
	main()
