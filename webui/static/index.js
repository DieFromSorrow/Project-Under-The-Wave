function classifyById() {
    const songId = document.getElementById('songId').value;
    document.getElementById('result').innerText = '处理中...';
    fetch(`/process/${songId}`, {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('result').innerHTML =
                `该歌曲的流派为： <strong>${data.genre_name.replaceAll('_', ' ')}</strong> `;
        } else {
            document.getElementById('result').innerText = `错误：${data.err}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = '请求失败，请稍后重试';
    });
}

function classifyByFile() {
    const file = document.getElementById('uploadFile').files[0];
    if (file) {
        const formData = new FormData();
        document.getElementById('result').innerText = '处理中...';
        formData.append('file', file);

        fetch('/process', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('result').innerHTML =
                    `该歌曲的流派为： <strong>${data.genre_name.replaceAll('_', ' ')}</strong> `;
            } else {
                document.getElementById('result').innerText = `错误：${data.err}`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerText = '请求失败，请稍后重试';
        });
    } else {
        document.getElementById('result').innerText = '请上传一个MP3文件';
    }
}
