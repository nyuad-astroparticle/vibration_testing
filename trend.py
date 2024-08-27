import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Range(object):

    def __init__(self, fRange:str, ampRange:str, low = 20, high = 2000):
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


    def find_range_index(self,freq_index):
        try:
            output = self.rangeIndixes[freq_index]
            # print(freq_index, 'belongs to frequency range #', output)
            return output
        except ValueError:
            return None  # If the frequency is not in the list

class CalibrationTrend(object):
    def __init__(self,
                 trendFilepath, 
                 fRange,
                 ampRange,
                 fLOW = 20, 
                 fHIGH = 300, 
                 stepHeight = 0.04,
                 ladderDrop = -0.8, 
                 NASAfilepath = './nasa.csv'):
        
        self.fLOW       = fLOW
        self.fHIGH      = fHIGH
        self.stepHeight = stepHeight
        self.ladderDrop = ladderDrop

        self.nasa_df            = self._load_nasa_df(NASAfilepath)
        self.trend              = self._load_trend(trendFilepath)
        self.validBrakePoints   = self._filter_trend()

        self.fRange = Range(fRange, ampRange, fLOW, fHIGH)

    def _load_nasa_df(self, filepath)               -> pd.DataFrame:
        return pd.read_csv(filepath)
    
    def _load_trend(self, filepath)                 -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df.Time = df.Time - df.Time[0]
        return df
    
    def plot_trend(self, toSave = False)            -> None:
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
        previous_break_point = df['Time'].iloc[0]

        for current_break_point in breakPoints:
            points_between = df[(df['Time'] > previous_break_point) & (df['Time'] < current_break_point)]
            if len(points_between) >= 2:
                validBreakPoints.append(previous_break_point)
                previous_break_point = current_break_point

        validBreakPoints.append(previous_break_point)
        # validBreakPoints.append(df['Time'].iloc[-1])

        return validBreakPoints

    def process_ladder(self, ladder, toPlot = False,
                        toSave = False)     -> list[tuple[float,float]]:

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
        step_dividers = ladder[abs(ladder['Slope']) > self.stepHeight]['Time'].values

        # Recalculate the valid dividers by ensuring all points on dividers are removed
        # Initialize a list to hold the valid dividers
        valid_dividers = []

        # Add a starting point for comparison
        previous_divider = ladder['Time'].iloc[0]

        for current_divider in step_dividers:
            points_between = ladder[(ladder['Time'] > previous_divider) & 
                                                (ladder['Time'] < current_divider)]
            if len(points_between) > 2:
                valid_dividers.append(previous_divider)
                previous_divider = current_divider

        # Append the last divider
        valid_dividers.append(previous_divider)
        
        # Filter the segment to remove points that coincide with divider positions
        filtered_segment = ladder[~ladder['Time'].isin(step_dividers)]
        filtered_segment = filtered_segment[~filtered_segment['Time'].isin(valid_dividers)]

        # Function to calculate the average amplitude within a segment
        def calculate_average_amplitude(start_time, end_time):
            segment = filtered_segment[(filtered_segment['Time'] >= start_time) & 
                                    (filtered_segment['Time'] < end_time)]
            return segment['Ampl'].mean()

        # Check if the difference in average amplitude between the first and second step is twice as large
        if len(valid_dividers) > 2:
            # Calculate the average amplitude for the first two steps
            first_step_avg = calculate_average_amplitude(valid_dividers[0], valid_dividers[1])
            second_step_avg = calculate_average_amplitude(valid_dividers[1], valid_dividers[2])

            # Calculate the differences in average amplitude for steps 2 onwards
            amplitude_differences = []
            for i in range(2, len(valid_dividers) - 1):
                avg_diff = abs(calculate_average_amplitude(valid_dividers[i], valid_dividers[i + 1]) -
                            calculate_average_amplitude(valid_dividers[i - 1], valid_dividers[i]))
                amplitude_differences.append(avg_diff)

            avg_diff_rest = np.mean(amplitude_differences)

            # Compare the first difference with twice the average of the remaining differences
            if abs(first_step_avg - second_step_avg) > 2 * avg_diff_rest:
                # Remove the data points corresponding to the zeroth step
                filtered_segment = filtered_segment[filtered_segment['Time'] >= valid_dividers[1]]
                # Remove the zeroth divider
                valid_dividers.pop(0)

        # Calculate geometric means and plot red Xs
        geometricMeans = []
        for i in range(len(valid_dividers) - 1):
            segment = filtered_segment[(filtered_segment['Time'] >= valid_dividers[i]) & 
                                        (filtered_segment['Time'] < valid_dividers[i + 1])]
            if len(segment) > 0:
                geom_mean_time = np.average(segment['Time'])
                geom_mean_ampl = np.average(segment['Ampl'])
                geometricMeans.append((geom_mean_time, geom_mean_ampl))

        if toPlot:
            # # Plot the filtered ladder segment with valid step dividers
            
            fig2 = plt.figure(figsize=(10, 6))  # Create a new figure object
            ax2 = fig2.add_subplot(111)  # Add a subplot (1x1 grid, first subplot)
            ax2.plot(ladder['Time'], ladder['Ampl'], label='Ampl')
            ax2.scatter(filtered_segment['Time'], filtered_segment['Ampl'], label='Ampl', marker='.', color='purple')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Filtered Ladder')
            ax2.legend()
            ax2.grid(True)

            # Add vertical dashed lines at valid step dividers
            for sd in valid_dividers:
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
        
        observed_steps = []
        for i in range(len(self.validBrakePoints) - 1):
            ladder = self.select_ladder(i)
            gm = self.process_ladder(ladder)
            observed_steps.append(len(gm))

        prescribed_steps = []

        for findex, f in enumerate(self.fRange.frequencyList):
            row = self.fRange.ampRange.iloc[self.fRange.find_range_index(findex)]
            rStart, rEnd, rStep = row['start'], row['end'], row['step']
            new_step = ((rEnd-rStart)//rStep + 1)
            prescribed_steps.append(new_step)

        if observed_steps != prescribed_steps:
            for i in range(len(observed_steps)):
                if observed_steps[i] == prescribed_steps[i]: continue
                print('Ladder', i, 'has', observed_steps[i], 'while should have', 
                          prescribed_steps[i])
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
        geometric_means = self.process_ladder(ladder)
    

        f = self.fRange.frequencyList[ladderIndex]
        g = self.nasa_df[self.nasa_df['frequency'] == f]['g'].iloc[0]
        observedAmpsInLadder = [gm[1] for gm in geometric_means]
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

    def create_amp_file(self, filename = None)      -> None:
        """
        This function write a csv file that contains the required mV amplitdue values
        for the actual shake test. Calibration done

        **Input**

        *filename* name of the output file 

        **Output**

        None

        """

        if filename is None:
            filename = 'SDD_calibration_' + str(self.fLOW) + '-' + str(self.fHIGH) + '.csv'
        final_amplitudes = []
        for i in range(len(self.validBrakePoints) - 1):
            final_amplitudes.append(self.amplitude_finder(i))
            self.nasa_df = self.nasa_df[self.nasa_df['frequency'].isin(self.fRange.frequencyList)]
        self.nasa_df['Amplitude (mV)'] = final_amplitudes
        # nasa_df = nasa_df.drop(columns=['a'])
        self.nasa_df.to_csv(filename, index=False)
    
    def examine_ladder(self, ladderIndex)           -> None:
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