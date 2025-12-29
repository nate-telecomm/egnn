package egnn
import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/floats"
	"math/rand"
	"math"
	"fmt"
	"encoding/gob"
	"bytes"
)

func addGaussianNoise(x *mat.Dense, std float64) {
	r, c := x.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x.Set(i, j, x.At(i, j)+rand.NormFloat64()*std)
		}
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func noisySigmoid(x float64, noiseStd float64) float64 {
    noise := rand.NormFloat64() * noiseStd
    return sigmoid(x + noise)
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)

			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}

		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, fmt.Errorf("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func (nn *NeuralNet) DumpNet() ([]byte, error) {
	var buffer bytes.Buffer
	encoder := gob.NewEncoder(&buffer)
	err := encoder.Encode(nn)
	if err != nil {
		return []byte{}, err
	}
	return buffer.Bytes(), nil
}

func LoadNet(b []byte) (*NeuralNet, error) {
	buffer := bytes.NewBuffer(b)
	decoder := gob.NewDecoder(buffer)

	var net NeuralNet
	err := decoder.Decode(&net)
	if err != nil {
		return nil, err
	}
	return &net, nil
}
