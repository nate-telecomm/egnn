package main
import (
	"math/rand"
	"fmt"
	"time"
	"gonum.org/v1/gonum/mat"
)

type NetConfig struct {
	InputNeurons   int
	OutputNeurons  int
	HiddenNeurons  int
	NumEpochs      int
	LearningRate   float64
}

type NeuralNet struct {
	config   NetConfig
	wHidden  *mat.Dense
	bHidden  *mat.Dense
	wOut     *mat.Dense
	bOut     *mat.Dense
}

func NewNet(conf NetConfig) *NeuralNet {
	return &NeuralNet{config: conf}
}

func (nn *NeuralNet) Train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.InputNeurons, nn.config.HiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.HiddenNeurons, nil)

	wOut := mat.NewDense(nn.config.HiddenNeurons, nn.config.OutputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.OutputNeurons, nil)


	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,

		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}

	}

	output := new(mat.Dense)

	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut

	nn.bOut = bOut

	return nil
}

func (nn *NeuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	for i := 0; i < nn.config.NumEpochs; i++ {
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)


		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)


		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)


		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.LearningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.LearningRate, bOutAdj)

		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.LearningRate, wHiddenAdj)

		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}
	return nil
}

func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, fmt.Errorf("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, fmt.Errorf("the supplied biases are empty")
	}

	output := new(mat.Dense)

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)

	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

type FeatureType int

const (
	Binary FeatureType = iota      // 0 or 1
	Continuous                      // float64, normalized
	Categorical                     // one-hot encoded
	Probability
)

type FeatureDefinition struct {
	Name       string
	Type       FeatureType
	Min        float64      // for Continuous
	Max        float64      // for Continuous
	Categories []string     // for Categorical
}
type OutputDefinition struct {
	Name string
	Type FeatureType
	Min  float64
	Max  float64
}

type NeuralInterface struct {
	InputSchema  []FeatureDefinition
	OutputSchema []OutputDefinition
}

func (ni *NeuralInterface) EncodeInput(input map[string]interface{}) *mat.Dense {
	features := make([]float64, 0)

	for _, def := range ni.InputSchema {
		value, exists := input[def.Name]

		if !exists || value == nil {
			switch def.Type {
			case Binary:
				features = append(features, 0.0) 
			case Continuous:
				features = append(features, (def.Min + def.Max) / 2)
			case Categorical:
				for i := range def.Categories {
					if i == 0 {
						features = append(features, 1.0)
					} else {
						features = append(features, 0.0)
					}
				}
			}
			continue
		}

		switch def.Type {
		case Binary:
			if value.(bool) {
				features = append(features, 1.0)
			} else {
				features = append(features, 0.0)
			}

		case Continuous:
			raw := value.(float64)
			normalized := (raw - def.Min) / (def.Max - def.Min)
			features = append(features, normalized)

		case Categorical:
			category := value.(string)
			for _, cat := range def.Categories {
				if cat == category {
					features = append(features, 1.0)
				} else {
					features = append(features, 0.0)
				}
			}
		}
	}

	return mat.NewDense(1, len(features), features)
}

func (ni *NeuralInterface) EncodeOutput(output map[string]float64) *mat.Dense {
	features := make([]float64, 0)

	for _, def := range ni.OutputSchema {
		value := output[def.Name]
		features = append(features, value)
	}

	return mat.NewDense(1, len(features), features)
}

func (ni *NeuralInterface) Decode(output *mat.Dense) map[string]float64 {
	decisions := make(map[string]float64)

	for i, def := range ni.OutputSchema {
		value := output.At(0, i) 

		switch def.Type {
		case Probability:
			decisions[def.Name] = value
		case Continuous:
			actual := value*(def.Max-def.Min) + def.Min
			decisions[def.Name] = actual
		}
	} 
	return decisions
}

type TrainingDatum struct {
	Inputs map[string]interface{}
	Outputs map[string]float64
}
