from Summary import summary

log_dir = "FOO"

"""
def test_writer():
    file_writer = summary.FileWriter(log_dir)
    file_writer.add_summary(1,2,3)
test_writer()

"""

def test_reader():
    file_reader = summary.FileReader(log_dir)
    file_reader.plot()
test_reader()