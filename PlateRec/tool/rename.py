import os


class rename_tool:
    @staticmethod
    def rename(src_name, target_name):
        os.rename(src_name, target_name)


if __name__ == '__main__':

    materials = os.listdir(os.path.abspath('../../Training'))
    for i in range(len(materials)):
        src = os.path.abspath('../../Training') + os.path.sep + materials[i]
        tgt = os.path.abspath('../../Training') + os.path.sep + materials[i][1:]
        try:
            os.rename(src, tgt)
        except:
            pass
