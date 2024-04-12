import configparser


class Config:

    @staticmethod
    def get_config():
        config = configparser.ConfigParser()
        config.read("config.ini")
        return config
