from flask import Flask, render_template, request, jsonify
from utils import crawl_data_by_id, get_pretrained_model, get_device, process, get_first_row, load_bytes_data, \
    load_from_path


app = Flask(__name__)
model = get_pretrained_model(params_pth='./checkpoints/version1/2024-5-15-epoch10.pt',
                             device=get_device())
genre_list = get_first_row(file_path='./data/v2/track_lists.csv')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process/<song_id>', methods=['GET'])
def process_by_id(song_id):
    try:
        data_obj = crawl_data_by_id(song_id)
    except ValueError as e:
        return jsonify({'success': False, 'err': 'This song is not available.'})
    else:
        waveform_mono, _ = load_bytes_data(data_obj)
        genre_name, max_idx, _ = process(waveform_mono, model, genre_list)
        return jsonify({'success': True, 'genre_name': genre_name})


@app.route('/process', methods=['POST'])
def process_data():
    if 'file' not in request.files:
        return jsonify({'success': False, 'err': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'err': 'No selected file'})
    try:
        waveform_mono, sample_rate = load_bytes_data(file.read())
        genre_name, max_idx, _ = process(waveform_mono, model, genre_list)
        return jsonify({'success': True, 'genre_name': genre_name})
    except Exception as e:
        return jsonify({'success': False, 'err': str(e)})
