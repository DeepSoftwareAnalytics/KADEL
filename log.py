import logging, time, sys, os, traceback

# record running info
class log:
    def __init__(self, logger_name=None, log_level=logging.DEBUG):

        if logger_name is None:
            logger_name = os.path.abspath(os.path.abspath(sys.argv[0]))
        self.logger = logging.getLogger(f"{logger_name}")
        self.logger.setLevel(log_level)
        
        self.last_modified_time = time.strftime("%Y%m%d_%H%M%S", time.gmtime(os.path.getmtime(sys.argv[0])))
        self.log_time = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(os.path.dirname(__file__), "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, f"{sys.argv[0]}_{self.last_modified_time}_{self.log_time}.log")

        fh = logging.FileHandler(self.log_file_path, encoding='utf-8')
        fh.setLevel(log_level)
        formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
        fh.setFormatter(formatter)


        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        fh.close()
        ch.close()

    def get_log_obj(self):
        return self.logger