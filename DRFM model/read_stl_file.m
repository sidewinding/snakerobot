function [F, V, N] = read_stl_file(name)
% Function for reading a stl file
% [F, v, N ] = read_stl_file('myfile.stl');
%
% where:
%
% F -> is a matrix of "faces"
% V -> is a matrix of "vertices"
% N -> is a matrix of "normals"
%
% "myfile.stl" is the file in stl ASCII format. It can be generated in any
% CAD software.
%
% Copyright, Lei Huang.
% 2022
%

% open file for reading
f = fopen(name, 'r');

% Initializate data
vertex = [ 0 0 0 ];
count_vertex = 1;
count_faces  = 1;
count_normal = 1;
i = 1;
while (1)
    % read a line
    new_line = fgetl(f);
    % process the line
    index_line = 0;
    index_number = 0;
    while (1)
        % increase index line
        index_line = index_line + 1;
        if (index_line > numel(new_line))
            break
        end
        s = new_line(index_line);
        % check for words 'normal'  
        if ( (index_line+5)<=numel(new_line) ) && ( (s=='n')||(s=='N') )
            word_normal = strcmp(new_line(index_line:(index_line+5)),'normal') | ...
                strcmp(new_line(index_line:(index_line+5)),'NORMAL');
            if word_normal
                % increase index
                index_line = index_line + 6;
                % read normal
                normal = read_three_numbers(new_line(index_line:max(size(new_line))));
                % store normal
                N(count_normal,:) = normal;
                % increase normal counter
                count_normal = count_normal + 1;
                % exit process line
                break
            end
        end
        % check for the word 'vertex'
        if ( (index_line+5)<=numel(new_line) ) && ( (s=='v'||s=='V') )
            word_vertex = strcmp(new_line(index_line:(index_line+5)),'vertex') | ...
                strcmp(new_line(index_line:(index_line+5)),'VERTEX');
            if word_vertex
                % increase index
                index_line = index_line + 6;
                % read vertex
                vertex = read_three_numbers(new_line(index_line:max(size(new_line))));
                % store vertex
                V(count_vertex,:) = vertex;
                % check for a new face
                if mod(count_vertex,3)==0
                    % store face
                    F(count_faces,:) = [ count_vertex-2 count_vertex-1 count_vertex ];
                    % increas faces counter
                    count_faces = count_faces + 1;
                end
                % increase vertex counter
                count_vertex = count_vertex + 1;
                % exit process line
                break
            end
        end
    end
    % check for end of file
    if (feof(f) == 1)
        break
    end
end

fclose(f);


p = patch('Faces',F,'Vertices',V);
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
daspect([1 1 1]);
view(3)
xlabel('Axis X');
ylabel('Axis Y');
zlabel('Axis Z');
camlight; lighting phong
grid on

return

% Function read three numbers
function numbers = read_three_numbers(buffer)

numbers = [ 0 0 ];
% process buffer
finish_number = 0;
start_number = 0;
reading_number = 0;
count_numbers = 0;
for (i = 1:max(size(buffer)))
    % a digit in a buffer
    s = buffer(i);
    % check for finish caracter 
    if ( s == ' ' | i == max(size(buffer)) )
        finish_number = i;
        % check for a number
        if (reading_number)&(finish_number>start_number)
            count_numbers = count_numbers + 1;
            numbers(1, count_numbers) = str2double(buffer(start_number:finish_number));
            if count_numbers > 3
                error('there are more than 3 numbers in a line')
            end
            reading_number = 0;
        end
    end
    % check for start caracter
    if reading_number
        if ~( (s=='+')|(s=='-')|(s=='.')|((s>='0')&(s<='9'))|(s=='E')|(s=='e') )
            error('there is a wrong caracter between digits')
        end
    else
        if ( (s=='+')|(s=='-')|(s=='.')|((s>='0')&(s<='9')) )
            start_number = i;
            reading_number = 1;
        end
    end

end

return
