import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Range(object):

    def __init__(self, fRange:str, ampRange:str, low:int = 20, high:int = 2000):
        self.fRange    = pd.read_csv(fRange)
        self.ampRange  = pd.read_csv(ampRange)
        self.frequencyList, self.rangeIndixes = self._expand_frequencies(
            self.fRange, low, high)

    def _expand_frequencies(self, dataFrame, low:int, high:int) -> tuple[
                list[int], list[int]]:
        """
        This internal function processes the range file for frequencies and returns
        an array of frequencies together with range index they belong to.

        **Input**

        *dataFrame* is the dataFrame containing the frequency ranges,

        *low* is the low frequency boundary,

        *high* is the upper frequency boundary, itself including.

        **Output**

        *frequencyList* is a list of all frequencies in order in the file

        *rangeIndexes* is a list of indexes describing a range a given freqency belongs to
        """
        df = dataFrame
        frequencList = []
        rangeIndixes = []
        
        # Expand each range and store the frequency with its corresponding range index
        for idx, row in df.iterrows():
            start, end, step = row['start'], row['end'], row['step']
            frequencies = list(range(start, end + 1, step))
            frequencies = [f for f in frequencies if (f >= low) & (f <= high)]
            frequencList.extend(frequencies)
            rangeIndixes.extend([idx] * len(frequencies))
        
        return frequencList, rangeIndixes


    def find_range_index(self,freqIndex:int):
        try:
            output = self.rangeIndixes[freqIndex]
            return output
        except ValueError:
            return None  # If the frequency is not in the list

class CalibrationTrend(object):
    def __init__(self,
                 trendFilepath : str, 
                 fRange                         : str,
                 ampRange                       : str,
                 fLOW                           : int       = 20, 
                 fHIGH                          : int       = 300, 
                 stepHeight                     : float     = 0.04,
                 ladderDrop                     : float     = -0.8, 
                 NASAfilepath                   : str       = './nasa.csv'):
        
        self.fLOW       = fLOW
        self.fHIGH      = fHIGH
        self.stepHeight = stepHeight
        self.ladderDrop = ladderDrop

        self.trend              = self._load_trend(trendFilepath)
        self.validBrakePoints   = self._filter_trend()

        self.fRange = Range(fRange, ampRange, fLOW, fHIGH)
        self.nasaDF            = self._load_nasaDF(NASAfilepath)
        self._filter_nasaDF()

    def _filter_nasaDF(self)                        -> None:
        """
        This function filter nasaDF frequencies keeping only the ones we use in our shake test.
        Modifies *nasaDF*

        **Input**

        None

        **Output**

        None
        """
        self.nasaDF = self.nasaDF[self.nasaDF['Frequency (Hz)'].isin(self.fRange.frequencyList)]

    def _load_nasaDF(self, filepath : str)          -> pd.DataFrame:
        return pd.read_csv(filepath)
    
    def _load_trend(self, filepath : str)           -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df.Time = df.Time - df.Time[0]
        return df
    
    def plot_trend(self, toSave : bool = False)     -> None:
        df = self.trend

        # # Plot Time vs Ampl with valid break points

        fig1 = plt.figure(figsize=(12, 6))  # Create a new figure object
        ax1 = fig1.add_subplot(111)  # Add a subplot (1x1 grid, first subplot)
        ax1.plot(df['Time'], df['Ampl'], label='Ampl', marker='.')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time vs Amplitude with Valid Break Points')
        ax1.legend()
        ax1.grid(True)

        # Add vertical dashed lines at valid break points
        for bp in self.validBrakePoints:
            ax1.axvline(x=bp, color='r', linestyle='--')

        plt.show()  # Show the figure


        if toSave: 
            fig1.savefig('Original Trend Curve')

    def _filter_trend(self)                         -> list[float]:
        """
        This function slices the trend curve into ladders.
        Each ladder corresponds to a given frequncy in the frequency list.

        **Input**

        None

        **Output**

        *validBreakPoints* are the slices. They surround each ladder on both sides.
        
        """
        
        df = self.trend
        df['Ampl_diff'] = df['Ampl'].diff()

        # Find time values where the drop in amplitude is more than ladder drop
        breakPoints = df[df['Ampl_diff'] < self.ladderDrop]['Time'].values
        
        validBreakPoints = []
        previousBreakPoint = df['Time'].iloc[0]

        for currentBreakPoint in breakPoints:
            pointsBetween = df[(df['Time'] > previousBreakPoint) & (df['Time'] < currentBreakPoint)]
            if len(pointsBetween) >= 2:
                validBreakPoints.append(previousBreakPoint)
                previousBreakPoint = currentBreakPoint

        validBreakPoints.append(previousBreakPoint)
        # validBreakPoints.append(df['Time'].iloc[-1])

        return validBreakPoints

    def process_ladder(self, ladder : pd.DataFrame, toPlot : bool = False,
                        toSave : bool = False)      -> list[tuple[float,float]]:

        """
        This function takes a ladder and splits into steps.
        A step corresponds to a given amplitude.

        **Input**

        *ladder* is a dataframe segment that contains data corresponding
        to one frequency only

        *toPlot* 

        *toSave*

        **Output**

        *geometricMeans* is a list of tuples with x and y coordinates for geometric 
        mean position of a step of the ladder

        """
        ladder = ladder.copy()

        # Calculate the slope between consecutive points
        ladder['Slope'] = ladder['Ampl'].diff() / ladder['Time'].diff()

        # Identify step dividers where the slope is greater than 1
        stepDividers = ladder[abs(ladder['Slope']) > self.stepHeight]['Time'].values

        # Recalculate the valid dividers by ensuring all points on dividers are removed
        # Initialize a list to hold the valid dividers
        validDividers = []

        # Add a starting point for comparison
        previousDivider = ladder['Time'].iloc[0]

        for current_divider in stepDividers:
            pointsBetween = ladder[(ladder['Time'] > previousDivider) & 
                                                (ladder['Time'] < current_divider)]
            if len(pointsBetween) > 2:
                validDividers.append(previousDivider)
                previousDivider = current_divider

        # Append the last divider
        validDividers.append(previousDivider)
        
        # Filter the segment to remove points that coincide with divider positions
        filteredSegment = ladder[~ladder['Time'].isin(stepDividers)]
        filteredSegment = filteredSegment[~filteredSegment['Time'].isin(validDividers)]

        # Function to calculate the average amplitude within a segment
        def calculate_average_amplitude(startTime : int, endTime : int):
            segment = filteredSegment[(filteredSegment['Time'] >= startTime) & 
                                    (filteredSegment['Time'] < endTime)]
            return segment['Ampl'].mean()

        # Check if the difference in average amplitude between the first and second step is twice as large
        if len(validDividers) > 2:
            # Calculate the average amplitude for the first two steps
            firstStepAvg = calculate_average_amplitude(validDividers[0], validDividers[1])
            secondStepAvg = calculate_average_amplitude(validDividers[1], validDividers[2])

            # Calculate the differences in average amplitude for steps 2 onwards
            amplitudeDifferences = []
            for i in range(2, len(validDividers) - 1):
                avg_diff = abs(calculate_average_amplitude(validDividers[i], validDividers[i + 1]) -
                            calculate_average_amplitude(validDividers[i - 1], validDividers[i]))
                amplitudeDifferences.append(avg_diff)

            avgDiffRest = np.mean(amplitudeDifferences)

            # Compare the first difference with twice the average of the remaining differences
            if abs(firstStepAvg - secondStepAvg) > 2 * avgDiffRest:
                # Remove the data points corresponding to the zeroth step
                filteredSegment = filteredSegment[filteredSegment['Time'] >= validDividers[1]]
                # Remove the zeroth divider
                validDividers.pop(0)

        # Calculate geometric means and plot red Xs
        geometricMeans = []
        for i in range(len(validDividers) - 1):
            segment = filteredSegment[(filteredSegment['Time'] >= validDividers[i]) & 
                                        (filteredSegment['Time'] < validDividers[i + 1])]
            if len(segment) > 0:
                geomMeanTime = np.average(segment['Time'])
                geomMeanAmpl = np.average(segment['Ampl'])
                geometricMeans.append((geomMeanTime, geomMeanAmpl))

        if toPlot:
            # # Plot the filtered ladder segment with valid step dividers

            fig2 = plt.figure(figsize=(10, 6))  # Create a new figure object
            ax2 = fig2.add_subplot(111)  # Add a subplot (1x1 grid, first subplot)
            ax2.plot(ladder['Time'], ladder['Ampl'], label='Ampl')
            ax2.scatter(filteredSegment['Time'], filteredSegment['Ampl'], label='Ampl', marker='.', color='purple')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Filtered Ladder')
            ax2.legend()
            ax2.grid(True)

            # Add vertical dashed lines at valid step dividers
            for sd in validDividers:
                ax2.axvline(x=sd, color='g', linestyle='--')

            # Plot red X's at the geometric mean points
            for gm in geometricMeans:
                ax2.scatter(gm[0], gm[1], color='red', marker='x', s=100)  # Adjust 's' for size of the X

            plt.show()  # Show the figure


        if toSave: 
            fig2.savefig('Ladder')

        return geometricMeans

    def select_ladder(self, ladderIndex:int = 0)    -> None | pd.DataFrame:
        """
        This function takes in a ladder index and return the dataframe for that ladder

        **Input**

        *laddderIndex* 

        **Output**

        *ladder* a dataframe
        """
        df = self.trend
        if ladderIndex == len(self.validBrakePoints):
            print('Last index or breakpoints should not be counted')
            return None
        ladder = df[(df['Time'] >= self.validBrakePoints[ladderIndex]) 
                    & (df['Time'] <= self.validBrakePoints[ladderIndex + 1])]
        
        return ladder

    def sanity_check(self)                          -> bool:
        """
        This function checks if what we have programmed into Key Sight has indeed 
        been executed and recorded properly.

        **Input**

        None

        **Output**
        
        Prints erroneous ladders and steps

        *bool* 
        """
        #Check if number of frequencies observed and expected are the same
        if len(self.fRange.frequencyList) != (len(self.validBrakePoints) - 1):
            print('Number of frequencies is off')
            return False

        #Check if each ladder has right number of steps i.e. each frequency has the right
        #number of amplitudes
        
        observedSteps = []
        for i in range(len(self.validBrakePoints) - 1):
            ladder = self.select_ladder(i)
            gm = self.process_ladder(ladder)
            observedSteps.append(len(gm))

        prescribedSteps = []

        for findex, f in enumerate(self.fRange.frequencyList):
            row = self.fRange.ampRange.iloc[self.fRange.find_range_index(findex)]
            rStart, rEnd, rStep = row['start'], row['end'], row['step']
            newStep = ((rEnd-rStart)//rStep + 1)
            prescribedSteps.append(newStep)

        if observedSteps != prescribedSteps:
            for i in range(len(observedSteps)):
                if observedSteps[i] == prescribedSteps[i]: continue
                print('Ladder', i, 'has', observedSteps[i], 'while should have', 
                          prescribedSteps[i])
            return False    

        return True            

    def amplitude_finder(self,ladderIndex:int = 0)  -> None | int:
        """
        This function reconciles the observed amplitude in g and prescribed amplitude in mV
        to Key Sight

        **Input**

        *ladderIndex* index of a ladder (or frequency) in question

        **Output**

        *finalAmplitudeInmV* returns the prescribed amplitude in mV that should produce
        the desired g value. In case it fails to find it returns None

        """
        ladder = self.select_ladder(ladderIndex)
        geometricMeans = self.process_ladder(ladder)
    

        f = self.fRange.frequencyList[ladderIndex]
        g = self.nasaDF[self.nasaDF['Frequency (Hz)'] == f]['g'].iloc[0]
        observedAmpsInLadder = [gm[1] for gm in geometricMeans]
        row = self.fRange.ampRange.iloc[self.fRange.find_range_index(ladderIndex)]
        rStart, rEnd, rStep = row['start'], row['end'], row['step']
        prescribedAmplitudes = [i for i in range(rStart, rEnd + 1, rStep)]

        for index,amp in enumerate(observedAmpsInLadder):
            if index == len(observedAmpsInLadder) -1: break
            if abs(g-amp) < abs(g-observedAmpsInLadder[index+1]): break
        if abs(g-amp) > 0.5:
            print('Failed to find ampilitude value')
            return None
        else:
            finalAmplitudeInmV = prescribedAmplitudes[index]
            return finalAmplitudeInmV

    def create_amp_file(self, filename:str = None)  -> None:
        """
        This function write a csv file that contains the required mV amplitdue values
        for the actual shake test. Calibration done

        **Input**

        *filename* name of the output file 

        **Output**

        None

        """
        
        # Createa the csv file of amp values for future use
        if filename is None:
            filename = 'SDD_calibration_' + str(self.fLOW) + '-' + str(self.fHIGH) + '.csv'
        finalAmplitudes = []
        for i in range(len(self.validBrakePoints) - 1):
            finalAmplitudes.append(self.amplitude_finder(i))
            # self.nasaDF = self.nasaDF[self.nasaDF['Frequency (Hz)'].isin(self.fRange.frequencyList)]
        self.nasaDF['Amplitude (mV)'] = finalAmplitudes

        outputDir = "output_csv"
        outputFilePath = os.path.join(outputDir, filename)

        # Check if the directory exists, if not, create it
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        self.nasaDF.to_csv(outputFilePath, index=False)

        return None
    
    def create_key_sight_file(self, filename : None | str = None,
                data : None | pd.DataFrame = None)  -> None:

        """
        This function write a key sight command file ready to use in the actual shake test

        **Input**

        *filename* If None given uses a default one

        **Output**

        None
        """

        if data is None:
            data = self.nasaDF

        # Createa the csv file of amp values for future use
        if filename is None:
            filename = 'SDD_calibration_' + str(self.fLOW) + '-' + str(self.fHIGH) + '_commands.txt'
        
        # Create the commands file to use in Key Sight

        commands = []
        commands.append('(Connect "33621A", "USB0::0x0957::0x5407::MY53700452::0::INSTR", "33500B/33600A Series Function / Arbitrary Waveform Generators / 2.09")')
        
        for index, row in data.iterrows():
            restingFrequency = 1
            frequency = row['Frequency (Hz)']
            voltage = row['Amplitude (mV)'] / 1000  # Convert mV to V
            
            # Main command
            commands.append(f":SOURce:APPLy:SINusoid {frequency},{voltage:.4f}")
            commands.append("(Wait 20000ms)")
            
            # Intermediate command with 0.001 V
            commands.append(f":SOURce:APPLy:SINusoid {restingFrequency},0.001")
            commands.append("(Wait 210000ms)")
        
        commands =  "\n".join(commands)

        # Save the commands to a text file

        outputDir = "output_to_key_sight"
        outputFilePath = os.path.join(outputDir, filename)

        # Check if the directory exists, if not, create it
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        with open(outputFilePath, 'w') as file:
            file.write(commands)

        return None
    
    def examine_ladder(self, ladderIndex : int)     -> None:
        """
        This function plots an individual ladder, already processed.

        **Input**

        *ladderIndex*

        **Output**

        None
        """

        ladder = self.select_ladder(ladderIndex)
        self.process_ladder(ladder, toPlot = True)
        return None        
    
class SecondaryCalibration(CalibrationTrend):
    def __init__(self,
                 trendFilepath                  : str, 
                 ampFile                        : str, 
                 fRange                         : str,
                 ampRange                       : str,
                 fLOW                           : int       = 20, 
                 fHIGH                          : int       = 300, 
                 stepHeight                     : float     = 0.04,
                 ladderDrop                     : float     = -0.8, 
                 NASAfilepath                   : str       = './nasa.csv'):
        super().__init__(
                 trendFilepath, 
                 fRange,
                 ampRange,
                 fLOW, 
                 fHIGH, 
                 stepHeight,
                 ladderDrop, 
                 NASAfilepath)
        
        self.ampFile = pd.read_csv(ampFile)
        self.frequencyList = self.ampFile['Frequency (Hz)']
        self._filter_nasaDF_secondary()

    def _filter_nasaDF_secondary(self)              -> None:
        """
        This is a secondary internal filter function that filters the nasaDF according
        to the ampFile which is the pre keysight command csv file

        **Input**

        None

        **Output**

        None

        """
        self.nasaDF = self.nasaDF[self.nasaDF['Frequency (Hz)'].isin(self.frequencyList)]
        return None

    def process_test(self, toPlot : bool = False)   -> list[tuple[float, float]]:
        """
        This function treats the whole shake test trend curve as a ladder returning the geometric
        average position of each frequency and amplitude pair

        **Input**

        *toPlot* is an internal argument so that this function can be used by plot_trend. If you
        want to plot call plot_trend instead.

        **Output**
        
        *processedLadder* is a list[tuple[float, float]] that containt coordinates of each frequency - 
        amplitude pair recorded

        """
        processedLadder = super().process_ladder(self.trend, toPlot)
        return processedLadder
    
    def plot_trend(self)                            -> None:
        """
        This fucntion is the same same as process_test, but it plots and returns nothing

        **Input**

        None

        **Output**

        None
        
        """
        self.process_test(toPlot = True)
        return None
    
    def sanity_check(self)                          -> bool:
        """
        This fucntion overrides the partent class sanity check, here it checks if the 
        number of frequencies observed is the same as the ones given to keysight

        **Input**

        None

        **Output**

        *bool* 
        """
        return len(self.process_test()) == len(self.frequencyList)
    
    def correct_amplitudes(self, filename: None | 
                           str = None)              -> pd.DataFrame:
        """
        This function is basically the essense of this class: it produces the new iteration of 
        the amp file that is supposedly closer to desired nasa g values. It takes the first shake test 
        result and the keysight amplitude values to adjust them hoping they converge to nasa g values. 

        **Input** 

        *filename* if None given, assigng a default name

        **Output**

        *data* is a pandas DataFrame. it saves it into a file and returns it as well
        """

        #Getting the parameters
        gm = self.process_test()
        measuredG = np.array([i[1] for i in gm])
        keySightAmp = self.ampFile['Amplitude (mV)']
        nasaG = np.array(self.nasaDF['g'])

        #The correction step
        #If the value of amplitude was much larger than the nasa one it brings it down
        #and vice versa. 
        correctedKeySightAmp = round(keySightAmp * (2 - (measuredG/nasaG)**1),0)

        #Nice formatting
        correctedKeySightAmp = [int(i) for i in correctedKeySightAmp]
        data = {'Frequency (Hz)':self.frequencyList, 'Amplitude (mV)':correctedKeySightAmp}
        data = pd.DataFrame(data)

        if filename is None:
            filename = 'Shake_test_next_iteration.csv'

        outputDir = "output_csv"
        outputFilePath = os.path.join(outputDir, filename)

        # Check if the directory exists, if not, create it
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        data.to_csv(outputFilePath, index=False)
        return data
    
    def create_key_sight_file(self, filename: None |
                               str = None)          -> None:
        if filename is None:
            filename = 'Shake_test_next_iteration_commands.txt'
        data = self.correct_amplitudes()
        return super().create_key_sight_file(filename, data)