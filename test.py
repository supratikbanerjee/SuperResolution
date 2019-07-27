import argparse, time, os

def main(config):
	print('test')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, help='Path to config YAML file.')
	args = parser.parse_args()
	with open(args.config, 'r') as stream:
	    config = yaml.safe_load(stream)
	main(config)
