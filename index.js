const express = require('express')
const multer = require('multer')
const jpeg = require('jpeg-js')

const tf = require('@tensorflow/tfjs-node')
const nsfw = require('nsfwjs')

const app = express()
const upload = multer()

let _model

app.all('*',(req,res,next)=>{
  // Add all the necessary headers and call next()
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
})
app.get("/",(req,res)=>res.send("Hello World!"))

const convert = async (img) => {
  // Decoded image in UInt8 Byte array
  const image = await jpeg.decode(img, true)

  const numChannels = 3
  const numPixels = image.width * image.height
  const values = new Int32Array(numPixels * numChannels)

  for (let i = 0; i < numPixels; i++)
    for (let c = 0; c < numChannels; ++c)
      values[i * numChannels + c] = image.data[i * 4 + c]

  return tf.tensor3d(values, [image.height, image.width, numChannels], 'int32')
}

app.post('/nsfw', upload.single("image"), async (req, res) => {
  if (!req.file)
    res.status(400).send("Missing image multipart/form-data")
  else {
    try{
    const image = await convert(req.file.buffer)
    const predictions = await _model.classify(image)
    image.dispose()
        const probabilities = predictions.map((item) => ({...item,probability:item.probability * 100}));
        function checkIfNsfw(values) {
        return values.some((value) => value.probability > 50);
        }
        const isNsfw = checkIfNsfw(probabilities);

        return res.status(200).send({ isNsfw, probabilities });
    res.json(predictions)
    }catch(err){
      console.error(err)
      return res.send("Internal server error")
    }
  }
})

const load_model = async () => {
  _model = await nsfw.load()
}

// Keep the model in memory, make sure it's loaded only once
load_model().then(() => app.listen(8080))
