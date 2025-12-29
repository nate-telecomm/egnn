package egnn
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
	Config   NetConfig
	WHidden  *mat.Dense
	BHidden  *mat.Dense
	WOut     *mat.Dense
	BOut     *mat.Dense
}

func NewNet(conf NetConfig) *NeuralNet {
        nn := &NeuralNet{Config: conf}

        randSource := rand.NewSource(time.Now().UnixNano())
        randGen := rand.New(randSource)

        nn.WHidden = mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, nil)
        nn.BHidden = mat.NewDense(1, nn.Config.HiddenNeurons, nil)
        nn.WOut = mat.NewDense(nn.Config.HiddenNeurons, nn.Config.OutputNeurons, nil)
        nn.BOut = mat.NewDense(1, nn.Config.OutputNeurons, nil)

        for _, param := range []*mat.Dense{nn.WHidden, nn.BHidden, nn.WOut, nn.BOut} {
                raw := param.RawMatrix().Data
                for i := range raw {
                        raw[i] = randGen.Float64()
                }
        }
        return nn
}

func (nn *NeuralNet) Train(x, y *mat.Dense) error {
        output := new(mat.Dense)

        if err := nn.backpropagate(x, y, nn.WHidden, nn.BHidden, nn.WOut, nn.BOut, output); err != nil {
                return err
        }

        return nil
}

func (nn *NeuralNet) backpropagate(x, y, WHidden, BHidden, WOut, BOut, output *mat.Dense) error {
        hiddenLayerInput := new(mat.Dense)
        hiddenLayerInput.Mul(x, WHidden)
        addBHidden := func(_, col int, v float64) float64 { return v + BHidden.At(0, col) }
        hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

        hiddenLayerActivations := new(mat.Dense)

        applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }

        hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

        outputLayerInput := new(mat.Dense)
        outputLayerInput.Mul(hiddenLayerActivations, WOut)
        addBOut := func(_, col int, v float64) float64 { return v + BOut.At(0, col) }
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
        errorAtHiddenLayer.Mul(dOutput, WOut.T())

        dHiddenLayer := new(mat.Dense)
        dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)


        WOutAdj := new(mat.Dense)
        WOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
        WOutAdj.Scale(nn.Config.LearningRate, WOutAdj)
        WOut.Add(WOut, WOutAdj)

        BOutAdj, err := sumAlongAxis(0, dOutput)
        if err != nil {
                return err
        }
        BOutAdj.Scale(nn.Config.LearningRate, BOutAdj)

        BOut.Add(BOut, BOutAdj)

        WHiddenAdj := new(mat.Dense)
        WHiddenAdj.Mul(x.T(), dHiddenLayer)
        WHiddenAdj.Scale(nn.Config.LearningRate, WHiddenAdj)

        WHidden.Add(WHidden, WHiddenAdj)

        BHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
        if err != nil {
                return err
        }
        BHiddenAdj.Scale(nn.Config.LearningRate, BHiddenAdj)
        BHidden.Add(BHidden, BHiddenAdj)
        return nil
}

func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.WHidden == nil || nn.WOut == nil {
		return nil, fmt.Errorf("the supplied weights are empty")
	}
	if nn.BHidden == nil || nn.BOut == nil {
		return nil, fmt.Errorf("the supplied biases are empty")
	}

	output := new(mat.Dense)

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.WHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.BHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.WOut)

	addBOut := func(_, col int, v float64) float64 { return v + nn.BOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	addGaussianNoise(output, 0.01)
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
