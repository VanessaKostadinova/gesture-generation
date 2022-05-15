import configparser

config = configparser.ConfigParser()
configPath = "C:\\Users\\vanes\\PycharmProjects\\gesture-generation\\config.ini"
config.read(configPath)
paths = config["DEFAULT"]
